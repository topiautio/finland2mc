"""Turn National Land Survey open data into 1 m grids for one area.

Output (in the work directory): *.npy grids, trees.npy, buildings.json and meta.json.
"""
import json
import os

import laspy
import numpy as np
import pyogrio.raw as pr
import rasterio
import shapely
from rasterio import features
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from scipy.interpolate import griddata

# Landcover codes used by the world generator
LC = dict(forest=0, field=1, garden=2, meadow=3, park=4, sports=5, cemetery=6, traffic=7,
          rock=8, rocky=9, quarry=10, bog_open=11, bog_forest=12, paludified=13,
          open_forest=14, storage=15)
# Burn order matters: later entries win.
LANDCOVER_CLASSES = [
    ("open_forest", "n", [39110]),
    ("paludified", "m", [35300]),
    ("bog_forest", "m", [35412, 35422]),
    ("bog_open", "m", [35411, 35421]),
    ("meadow", "m", [32800]),
    ("field", "m", [32611]),
    ("garden", "m", [32612]),
    ("park", "m", [32900]),
    ("sports", "m", [33100]),
    ("cemetery", "m", [32200]),
    ("quarry", "m", [32500, 32111, 32112]),
    ("storage", "n", [38900]),
    ("rocky", "m", [34700]),
    ("rock", "m", [34100]),
    ("traffic", "m", [32421]),
]
WATER_CLASSES = [36200, 36211, 36313, 44300]  # lake, sea, river area, basin
SEA = 36211
# Road raster codes: (topographic class codes, full width in metres)
ROADS = {
    1: ([12111, 12112, 12121, 12122], None),  # main roads, width per class below
    2: ([12131, 12132], None),
    3: ([12141], 4.0),  # roadway
    4: ([12316], 3.0),  # track
    5: ([12314], 3.0),  # pedestrian and bicycle route
    6: ([12313], 1.5),  # path
    7: ([14111, 14112, 14121, 14131], 5.0),  # railway, tram
}
ROAD_WIDTH = {12111: 12, 12112: 11, 12121: 9, 12122: 7.5, 12131: 6, 12132: 5}
ROAD_ORDER = [6, 4, 5, 3, 2, 1, 7]
BUILDING_KIND = {
    42211: "house", 42212: "apartment", 42221: "public", 42222: "public_tall",
    42231: "cottage", 42232: "cottage", 42241: "industrial", 42242: "industrial_tall",
    42251: "church", 42252: "church", 42270: "church", 42261: "shed", 42262: "public",
}
WORLD_BOTTOM, WORLD_TOP = -60, 316  # keep a little room inside Minecraft's -64..319


def height_rule(offset):
    """How Minecraft y relates to height above sea level, e.g. 'metres above sea level + 62'."""
    return f"metres above sea level {'-' if offset >= 0 else '+'} {abs(offset)}"


class Prep:
    def __init__(self, area, files, workdir, log=print):
        self.area, self.files, self.out, self.log = area, files, workdir, log
        self.size = area.size
        os.makedirs(workdir, exist_ok=True)

    # --- helpers -------------------------------------------------------------------------------
    def read_layer(self, prefix, geomtype, codes, cols=("LUOKKA",)):
        """Features of the given classes from all map sheets: (geoms, {field: array})."""
        geoms, attrs = [], {c: [] for c in cols}
        for d in self.files["mtk"]:
            path = d / f"{prefix}_{d.name}_{geomtype}.shp"
            if not path.exists():
                continue
            meta, _, g, fl = pr.read(path, columns=list(cols), bbox=self.area.bounds)
            if g is None or len(g) == 0:
                continue
            data = {n: np.asarray(v) for n, v in zip(meta["fields"], fl)}
            keep = np.isin(data["LUOKKA"].astype(int), codes)
            geoms += list(shapely.from_wkb(np.asarray(g, dtype=object)[keep]))
            for c in cols:
                attrs[c] += list(data[c][keep])
        return geoms, {c: np.asarray(v) for c, v in attrs.items()}

    def rasterize(self, shapes, dtype=np.uint8, fill=0):
        shapes = list(shapes)
        if not shapes:
            return np.full((self.size, self.size), fill, dtype)
        return features.rasterize(shapes, out_shape=(self.size, self.size), transform=self.area.transform,
                                  fill=fill, dtype=dtype)

    def merge_across_sheets(self, geoms):
        """Polygons are cut at the 12 km map sheet edges; glue the pieces back together."""
        x0, y0, x1, y1 = self.area.bounds
        lines = [shapely.LineString([(x, y0), (x, y1)]) for x in range(8000 + (x0 - 8000) // 12000 * 12000, x1 + 1, 12000)]
        lines += [shapely.LineString([(x0, y), (x1, y)]) for y in range(6000 + (y0 - 6000) // 12000 * 12000, y1 + 1, 12000)]
        if not lines:
            return geoms
        edges = shapely.union_all(lines).buffer(0.6)
        touching = [g for g in geoms if g.intersects(edges)]
        rest = [g for g in geoms if not g.intersects(edges)]
        if touching:
            rest += list(shapely.get_parts(shapely.union_all([g.buffer(0.05) for g in touching])))
        return rest

    # --- steps ---------------------------------------------------------------------------------
    def elevation(self):
        self.log("elevation")
        b = self.area.bounds
        dem = np.full((self.size, self.size), np.nan, np.float32)
        if self.files["dem"]:
            src = [rasterio.open(f) for f in self.files["dem"]]
            arr, tr = merge(src, bounds=(b[0] - 4, b[1] - 4, b[2] + 4, b[3] + 4), res=2, nodata=-9999)
            arr = np.where(arr[0] == -9999, np.nan, arr[0]).astype(np.float32)
            reproject(arr, dem, src_transform=tr, src_crs=src[0].crs, dst_transform=self.area.transform,
                      dst_crs=src[0].crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        missing = ~np.isfinite(dem)
        if missing.all():
            raise SystemExit("No elevation data for this area. Is it in Finland, and not out at sea?")
        if missing.any():
            # tiles that don't exist are open sea; anything else gets the nearest known height
            sea, _ = self.read_layer("m", "p", [SEA])
            sea = self.rasterize(sea) > 0
            dem[missing & sea] = 0.0
            still = ~np.isfinite(dem)
            if still.any():
                idx = ndimage.distance_transform_edt(still, return_distances=False, return_indices=True)
                dem = dem[tuple(idx)]
        return dem

    def lidar(self, dem):
        """Highest laser return per 2 m cell, the canopy height above ground, and where we have data."""
        self.log("laser scanning")
        n2 = self.size // 2
        b = self.area.bounds
        dsm = np.full(n2 * n2, -np.inf, np.float32)
        for f in self.files["lidar"]:
            with laspy.open(f) as reader:
                for pts in reader.chunk_iterator(4_000_000):
                    x, y = np.asarray(pts.x), np.asarray(pts.y)
                    keep = (np.asarray(pts.classification) != 7) & (x >= b[0]) & (x < b[2]) & (y > b[1]) & (y <= b[3])
                    if not keep.any():
                        continue
                    col = ((x[keep] - self.area.west) // 2).astype(np.int64)
                    row = ((self.area.north - y[keep]) // 2).astype(np.int64)
                    np.maximum.at(dsm, row * n2 + col, np.asarray(pts.z, dtype=np.float32)[keep])
            self.log(f"  {f.name}")
        dsm = dsm.reshape(n2, n2)
        dem2 = dem.reshape(n2, 2, n2, 2).mean(axis=(1, 3))
        has = np.isfinite(dsm)
        covered = ndimage.binary_closing(has, iterations=4) | has
        if has.any():
            idx = ndimage.distance_transform_edt(~has, return_distances=False, return_indices=True)
            dsm = dsm[tuple(idx)]
        else:
            dsm = dem2.copy()
        dsm = np.where(covered, dsm, dem2)
        chm = np.clip(dsm - dem2, 0, 45)
        self.log(f"  laser data covers {covered.mean() * 100:.0f}% of the area")
        return dsm, chm, covered

    def water(self, dem):
        self.log("water")
        geoms, attr = self.read_layer("m", "p", WATER_CLASSES)
        water = self.rasterize(geoms) > 0
        sea = self.rasterize(g for g, c in zip(geoms, attr["LUOKKA"]) if int(c) == SEA) > 0
        comp, ncomp = ndimage.label(water, structure=np.ones((3, 3)))
        level = np.zeros((self.size, self.size), np.float32)
        if ncomp:
            lv = ndimage.median(dem, comp, index=np.arange(1, ncomp + 1)).astype(np.float32)
            level[water] = lv[comp[water] - 1]
            # sea level is 0 m, whatever the elevation model says at the shore
            sea_comp = np.unique(comp[sea])
            level[np.isin(comp, sea_comp[sea_comp > 0])] = 0.0
        is_sea = np.isin(comp, np.unique(comp[sea])) & water if sea.any() else np.zeros_like(water)

        # Depth: contours (1.5 ... 25 m) and the shoreline at depth 0, interpolated on a 4 m grid
        contours, cattr = self.read_layer("k", "v", [54100], cols=("LUOKKA", "KORARV"))
        pts, vals = [], []
        for g, d in zip(contours, cattr["KORARV"]):
            c = shapely.get_coordinates(shapely.segmentize(g, 4))
            pts.append(c)
            vals.append(np.full(len(c), float(d)))
        shore = water & ~ndimage.binary_erosion(water)
        sr, sc = np.nonzero(shore)
        pts.append(np.c_[self.area.west + sc[::3] + 0.5, self.area.north - sr[::3] - 0.5])
        vals.append(np.zeros(len(sr[::3])))
        pts, vals = np.concatenate(pts), np.concatenate(vals)
        n4 = self.size // 4
        g4 = np.arange(n4) * 4 + 2
        gx, gy = np.meshgrid(self.area.west + g4, self.area.north - g4)
        w4 = water.reshape(n4, 4, n4, 4).any(axis=(1, 3))
        depth4 = np.zeros(w4.shape, np.float32)
        if w4.any() and len(vals) >= 3 and contours:
            depth4[w4] = griddata(pts, vals, (gx[w4], gy[w4]), method="linear", fill_value=np.nan)
        depth = ndimage.zoom(np.nan_to_num(depth4), 4, order=1)[:self.size, :self.size]
        dist = ndimage.distance_transform_edt(water)
        fallback = np.where(is_sea, np.minimum(1 + dist * 0.08, 10.0), np.minimum(1 + dist * 0.12, 3.0))
        depth = np.where(water, np.clip(np.maximum(depth, fallback), 1, 40), 0).astype(np.float32)
        self.log(f"  water bodies {ncomp}, deepest {depth.max():.0f} m" + (", sea" if sea.any() else ""))

        streams, sattr = self.read_layer("m", "v", [36311, 36312])
        stream = self.rasterize(g.buffer(0.7 if c == 36311 else 1.6) for g, c in zip(streams, sattr["LUOKKA"])) > 0
        return water, level, depth, stream & ~water

    def roads(self, dem, water, water_level):
        self.log("roads and bridges")
        codes = sorted(c for v in ROADS.values() for c in v[0])
        geoms, attr = self.read_layer("l", "v", codes, cols=("LUOKKA", "VERSUH"))
        cls = attr["LUOKKA"].astype(int)
        vers = np.nan_to_num(attr["VERSUH"].astype(float))
        width = np.array([ROAD_WIDTH.get(c) or next(v[1] for v in ROADS.values() if c in v[0]) for c in cls])
        rcode = np.array([next(k for k, v in ROADS.items() if c in v[0]) for c in cls])
        garr = np.asarray(geoms, dtype=object)
        road = np.zeros((self.size, self.size), np.uint8)
        for k in ROAD_ORDER:
            sel = (rcode == k) & (vers < 1)
            road[self.rasterize(g.buffer(w / 2, cap_style="flat") for g, w in zip(garr[sel], width[sel])) > 0] = k

        # Bridges (vertical position 1): deck height interpolated between the ends of the bridge line
        deck = np.full((self.size, self.size), -9999, np.int16)
        bridge_cls = np.zeros((self.size, self.size), np.uint8)
        pillar = np.zeros((self.size, self.size), bool)
        W, N, S = self.area.west, self.area.north, self.size
        dem_at = lambda x, y: float(dem[int(np.clip(N - y, 0, S - 1)), int(np.clip(x - W, 0, S - 1))])
        nb = 0
        for g, w, k, v in zip(geoms, width, rcode, vers):
            if v < 1:
                continue
            nb += 1
            c = shapely.get_coordinates(g)
            z0, z1 = dem_at(*c[0]), dem_at(*c[-1])
            rr, cc = np.nonzero(self.rasterize([g.buffer(w / 2 + 0.5, cap_style="flat")]))
            if not len(rr):
                continue
            px = shapely.points(W + cc + 0.5, N - rr - 0.5)
            t = shapely.line_locate_point(g, px, normalized=True)
            z = z0 + (z1 - z0) * t
            z = np.maximum(z, np.where(water[rr, cc], water_level[rr, cc], -1e9) + 2.5)
            deck[rr, cc] = np.maximum(deck[rr, cc], np.round(z).astype(np.int16))
            bridge_cls[rr, cc] = k
            along = t * g.length
            dl = shapely.distance(g, px)
            pillar[rr, cc] |= ((along % 14) < 1.5) & (dl < w / 2 - 0.5) & (along > 3) & (along < g.length - 3)
        self.log(f"  bridges {nb}")
        causeway = (road > 0) & water & (deck == -9999)  # roads over water without a bridge flag
        return road, deck, bridge_cls, pillar, causeway

    def landcover(self):
        self.log("land cover")
        lc = np.zeros((self.size, self.size), np.uint8)
        field_id = np.zeros((self.size, self.size), np.int32)
        for name, prefix, codes in LANDCOVER_CLASSES:
            geoms, _ = self.read_layer(prefix, "p", codes)
            if name == "field":
                ids = self.rasterize(((g, i + 1) for i, g in enumerate(self.merge_across_sheets(geoms))), np.int32)
                field_id = np.where(ids > 0, ids, field_id)
                lc[ids > 0] = LC[name]
            else:
                lc[self.rasterize(geoms) > 0] = LC[name]
        urban, _ = self.read_layer("u", "p", [40200])
        return lc, field_id, self.rasterize(urban) > 0

    def buildings(self, dem, dsm, covered):
        self.log("buildings")
        geoms, attr = self.read_layer("r", "p", list(BUILDING_KIND))
        pieces = list(zip(geoms, attr["LUOKKA"].astype(int)))
        tree = shapely.STRtree([g for g, _ in pieces]) if pieces else None
        blds = []
        for g in self.merge_across_sheets([g for g, _ in pieces]):
            if g.area < 6:
                continue
            hits = tree.query(g.representative_point(), predicate="intersects")
            blds.append((g, pieces[hits[0]][1] if len(hits) else 42261))
        bid = self.rasterize(((g, i + 1) for i, (g, _) in enumerate(blds)), np.int32)
        idx = np.arange(1, len(blds) + 1)
        table = []
        if blds:
            area = ndimage.sum(np.ones_like(bid, np.float32), bid, idx)
            ground = ndimage.median(dem, bid, idx)
            dsm1 = np.repeat(np.repeat(dsm, 2, 0), 2, 1)
            top = ndimage.labeled_comprehension(dsm1, bid, idx, lambda v: np.percentile(v, 85), np.float32, np.nan)
            cov1 = np.repeat(np.repeat(covered, 2, 0), 2, 1)
            with np.errstate(invalid="ignore"):  # buildings too small to cover a whole 1 m cell
                cov = ndimage.mean(cov1, bid, idx) > 0.5
            for i, (g, c) in enumerate(blds):
                h = float(top[i] - ground[i]) if cov[i] and np.isfinite(top[i]) else None
                table.append(dict(id=i + 1, cls=int(c), kind=BUILDING_KIND[c], area=float(area[i]),
                                  ground=float(ground[i]), lidar_h=h))
        # distance from the footprint edge (1 = wall ring); neighbouring buildings count as edges
        edge = np.zeros_like(bid, bool)
        for sh in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            edge |= np.roll(bid, sh, axis=(0, 1)) != bid
        edge &= bid > 0
        bdist = np.where(bid > 0, np.floor(ndimage.distance_transform_edt(~edge & (bid > 0))) + 1, 0)
        self.log(f"  buildings {len(blds)}")
        return bid, np.minimum(bdist, 255).astype(np.uint8), table

    def trees(self, chm, covered, dem, lc, water, road, bid, urban):
        """Tree tops from the canopy height model, plus made-up forest where there is no laser data."""
        self.log("trees")
        sm = ndimage.gaussian_filter(chm, 0.6)
        peak = (sm == ndimage.maximum_filter(sm, size=3)) & (chm >= 2.5) & covered
        # at ~1 point/m2 not every crown shows up as a peak; fill closed canopy at ~4 m spacing
        grid = np.zeros_like(peak)
        grid[::2, ::2] = True
        peak |= (sm > 4) & ~ndimage.binary_dilation(peak) & grid & covered
        r, c = np.nonzero(peak)
        rng = np.random.default_rng(42)
        h = chm[r, c]
        # no laser data: forest on forest land at ~5 m spacing with typical heights
        nolidar = ~covered & grid
        rr, cc = np.nonzero(nolidar)
        forest_land = np.isin(lc, [LC["forest"], LC["bog_forest"], LC["open_forest"], LC["paludified"]]) & ~urban
        keep = forest_land[rr * 2, cc * 2] & (rng.random(len(rr)) < 0.7)
        r, c = np.r_[r, rr[keep]], np.r_[c, cc[keep]]
        h = np.r_[h, rng.uniform(9, 22, keep.sum())]
        x = c * 2 + rng.integers(0, 2, len(c))
        z = r * 2 + rng.integers(0, 2, len(r))
        blocked = ndimage.binary_dilation(water | (road > 0) | (bid > 0) | (lc == LC["traffic"]))
        ok = ~blocked[z, x]
        x, z, h = x[ok], z[ok], h[ok]
        # species: pine on ridges, rock and bogs; spruce in hollows; birch near open land
        tpi = dem - ndimage.uniform_filter(dem, 61)
        open_land = np.isin(lc, [LC["field"], LC["meadow"], LC["garden"], LC["park"]]) | (bid > 0) | (road > 0)
        near_open = ndimage.binary_dilation(open_land, iterations=12)
        p = rng.random(len(x))
        t = tpi[z, x]
        pine = np.where(t > 4, 0.65, np.where(t < -4, 0.2, 0.4))
        pine = np.where(np.isin(lc[z, x], [LC["rock"], LC["rocky"], LC["bog_forest"], LC["bog_open"]]), 0.8, pine)
        birch = np.where(near_open[z, x], 0.4, 0.12)
        species = np.where(p < birch, 2, np.where(p < birch + (1 - birch) * pine, 1, 0))  # spruce, pine, birch
        self.log(f"  trees {len(x)}")
        return np.c_[x, z, np.round(h * 10), species].astype(np.int32)

    def run(self):
        dem = self.elevation()
        dsm, chm, covered = self.lidar(dem) if self.files["lidar"] else (
            dem.reshape(self.size // 2, 2, self.size // 2, 2).mean(axis=(1, 3)),
            np.zeros((self.size // 2,) * 2, np.float32), np.zeros((self.size // 2,) * 2, bool))
        water, water_level, depth, stream = self.water(dem)
        road, deck, bridge_cls, pillar, causeway = self.roads(dem, water, water_level)
        lc, field_id, urban = self.landcover()
        bid, bdist, btable = self.buildings(dem, dsm, covered)
        trees = self.trees(chm, covered, dem, lc, water & ~causeway, road, bid, urban)

        # Vertical placement: the most common water surface at y = 62 (vanilla sea level) if it fits
        if water.any():
            main = float(np.bincount(np.round(water_level[water]).astype(int).clip(0)).argmax())
        else:
            main = float(np.median(dem)) - 8
        lowest = float((dem - np.where(water, depth, 0)).min())
        # trees reach ~30 m; building tops come from the laser data (stray points in the sky are ignored)
        tops = [b["ground"] + min(b["lidar_h"], 150) for b in btable if b["lidar_h"]]
        highest = float(max([dem.max() + 30] + tops)) + 3
        offset = int(round(main)) - 62
        if lowest - offset < WORLD_BOTTOM or highest - offset > WORLD_TOP:
            if highest - lowest > WORLD_TOP - WORLD_BOTTOM:
                raise SystemExit(f"The height difference here ({highest - lowest:.0f} m) doesn't fit in a Minecraft "
                                 "world (384 blocks). Try a smaller area.")
            offset = int(np.ceil(highest - WORLD_TOP)) if highest - offset > WORLD_TOP else int(np.floor(lowest - WORLD_BOTTOM))
        self.log(f"  y = {height_rule(offset)}")

        ground = np.round(dem).astype(np.int16) - offset
        water_y = np.where(water, np.round(water_level).astype(np.int16) - offset, -9999).astype(np.int16)
        bottom_y = np.where(water, water_y - np.round(depth).astype(np.int16), ground).astype(np.int16)
        water_y[causeway] = -9999
        ground[causeway] = (np.round(water_level[causeway]) - offset + 1).astype(np.int16)
        water &= ~causeway
        # keep shores at least as high as the water next to them so lakes stay put
        near = ndimage.binary_dilation(water, iterations=2) & ~water
        if near.any():
            _, (ir, ic) = ndimage.distance_transform_edt(~water, return_indices=True)
            ground = np.where(near, np.maximum(ground, water_y[ir, ic]), ground)
        bottom_y = np.where(water, bottom_y, ground)
        deck_y = np.where(deck > -9999, deck - offset, -9999).astype(np.int16)

        grids = dict(ground=ground, water_y=water_y, bottom_y=bottom_y, stream=stream, road=road, deck_y=deck_y,
                     bridge_cls=bridge_cls, pillar=pillar, lc=lc, field_id=field_id, urban=urban, bid=bid,
                     bdist=bdist, trees=trees)
        for k, v in grids.items():
            np.save(os.path.join(self.out, f"{k}.npy"), v)
        for b in btable:
            b["floor_y"] = int(round(b["ground"])) - offset
        json.dump(btable, open(os.path.join(self.out, "buildings.json"), "w"))
        meta = dict(**self.area.to_dict(), X0=self.area.x0, SIZE=self.size, HALF=self.area.half, offset=offset)
        json.dump(meta, open(os.path.join(self.out, "meta.json"), "w"), indent=1)
        return meta
