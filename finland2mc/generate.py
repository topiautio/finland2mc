"""Build Minecraft 26.3 region files from the grids made by prepare.py."""
import json
import multiprocessing as mp
import os
import time

import numpy as np
from scipy import ndimage

from .mcformat import MIN_Y, HEIGHT, Blocks, encode_chunk, write_region

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NONE = -9999
GRIDS = ["ground", "water_y", "bottom_y", "stream", "road", "deck_y", "bridge_cls", "pillar",
         "lc", "field_id", "urban", "bid", "bdist", "trees"]
# Set by load() in the parent process; worker processes inherit them through fork.
X0 = SIZE = HALF = None
G = BUILDINGS = BT = NOISE = NOISE2 = TREES = TORDER = TKEY_SORTED = None

B = Blocks(os.path.join(DATA, "blocks_26.3.json"))
AIR = B("air")
STONE, DEEPSLATE, BEDROCK, DIRT = B("stone"), B("deepslate"), B("bedrock"), B("dirt")
GRASS, PODZOL, MOSS, COARSE = B("grass_block"), B("podzol"), B("moss_block"), B("coarse_dirt")
SAND, GRAVEL, CLAY, MUD = B("sand"), B("gravel"), B("clay"), B("mud")
WATER, FARMLAND, DIRT_PATH = B("water"), B("farmland"), B("dirt_path")
COBBLE, MOSSY_COBBLE, ANDESITE = B("cobblestone"), B("mossy_cobblestone"), B("andesite")
ASPHALT, PAVEMENT = B("gray_concrete"), B("polished_andesite")
FOUNDATION, GLASS = B("stone_bricks"), B("glass")
SLAB_STONE, SLAB_WOOD, PLANKS_DECK = B("stone_brick_slab[type=bottom]"), B("spruce_slab[type=bottom]"), B("spruce_planks")
LOG = {0: B("spruce_log"), 1: B("spruce_log"), 2: B("birch_log")}
LEAVES = {0: B("spruce_leaves[persistent=true,distance=1]"),
          1: B("oak_leaves[persistent=true,distance=1]"),  # pine crowns: lighter, biome-tinted green
          2: B("birch_leaves[persistent=true,distance=1]")}
SHORT_GRASS, FERN, BUSH, FIREFLY = B("short_grass"), B("fern"), B("bush"), B("firefly_bush")
DRY_GRASS = B("short_dry_grass")
FERN_LO, FERN_HI = B("large_fern[half=lower]"), B("large_fern[half=upper]")
TALL_LO, TALL_HI = B("tall_grass[half=lower]"), B("tall_grass[half=upper]")
FLOWERS = np.array([B("dandelion"), B("poppy"), B("oxeye_daisy"), B("cornflower")], np.uint16)
LITTER = np.array([B(f"leaf_litter[facing={f},segment_amount={n}]") for f in ("north", "east", "south", "west")
                   for n in (1, 2, 3, 4)], np.uint16)
DIRS = ("north", "south", "west", "east")

# --- building styles ---------------------------------------------------------------------------
ROOFS = [("deepslate_tiles", "deepslate_tile_stairs", 0.35), ("bricks", "brick_stairs", 0.2),
         ("dark_oak_planks", "dark_oak_stairs", 0.15), ("mud_bricks", "mud_brick_stairs", 0.1),
         ("polished_blackstone", "polished_blackstone_stairs", 0.2)]
ROOF_FULL = np.array([B(r[0]) for r in ROOFS], np.uint16)
ROOF_STAIRS = np.array([[B(f"{r[1]}[facing={d}]") for d in DIRS] for r in ROOFS], np.uint16)
WALLS = {
    "house": [("red_terracotta", .3), ("yellow_terracotta", .15), ("white_concrete", .2), ("brown_terracotta", .1),
              ("bricks", .1), ("spruce_planks", .1), ("light_gray_concrete", .05)],
    "cottage": [("red_terracotta", .4), ("spruce_planks", .3), ("brown_terracotta", .2), ("yellow_terracotta", .1)],
    "shed": [("red_terracotta", .6), ("spruce_planks", .3), ("light_gray_concrete", .1)],
    "apartment": [("white_concrete", .35), ("light_gray_concrete", .25), ("bricks", .25), ("yellow_terracotta", .15)],
    "public": [("bricks", .4), ("white_concrete", .3), ("light_gray_concrete", .3)],
    "industrial": [("light_gray_concrete", .4), ("white_concrete", .2), ("cyan_terracotta", .2), ("gray_concrete", .2)],
    "church": [("white_concrete", 1.0)],
}
STYLE = {  # kind: (wall palette, default wall height, flat roof?, hip roof cap)
    "house": ("house", 3, False, 6), "cottage": ("cottage", 3, False, 5), "shed": ("shed", 2, False, 4),
    "apartment": ("apartment", 12, True, 0), "public_tall": ("public", 12, True, 0),
    "public": ("public", 6, None, 5), "industrial": ("industrial", 7, None, 4),
    "industrial_tall": ("industrial", 10, True, 0), "church": ("church", 10, False, 30),
}


def building_table():
    n = len(BUILDINGS) + 1
    t = {k: np.zeros(n, np.int32) for k in ("floor", "hw", "flat", "cap", "wall", "roof", "floorblk", "winmod")}
    rng = np.random.default_rng(7)
    for b in BUILDINGS:
        i = b["id"]
        palette, hw, flat, cap = STYLE[b["kind"]]
        lh = b["lidar_h"]
        lh = lh if lh is not None and 2 <= lh <= 150 else None
        if flat is None:
            flat = b["area"] > 300
        if flat:
            hw = int(round(np.clip(lh, 3, 315 - b["floor_y"]))) if lh else hw
        elif b["kind"] in ("house", "cottage"):
            hw = 6 if lh and lh > 7.5 else 3
        elif b["kind"] == "shed":
            hw = 3 if lh and lh > 4.5 else 2
        elif b["kind"] == "church":
            hw = int(np.clip((lh or 16) * 0.55, 7, 16))
        else:
            hw = int(round(np.clip(lh - 2.5, 3, 12))) if lh else hw
        names, weights = zip(*WALLS[palette])
        t["floor"][i] = b["floor_y"]
        t["hw"][i] = hw
        t["flat"][i] = flat
        t["cap"][i] = cap
        t["wall"][i] = B(rng.choice(names, p=np.array(weights) / sum(weights)))
        t["roof"][i] = rng.choice(len(ROOFS), p=[r[2] for r in ROOFS])
        t["floorblk"][i] = B("polished_andesite" if flat else "oak_planks")
        t["winmod"][i] = 4 if palette == "industrial" else 3
    t["flatroof"] = np.where(np.isin(t["wall"], [B("light_gray_concrete")]), B("light_gray_concrete"), B("gray_concrete"))
    return t



def hash01(x, z, salt):
    h = (np.asarray(x, np.int64) * 374761393 + np.asarray(z, np.int64) * 668265263 + np.int64(salt) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h ^= h >> 16
    return (h & 0xFFFFFF) / float(0x1000000)


def smooth_noise(seed, cell):
    rng = np.random.default_rng(seed)
    n = SIZE // cell + 2
    small = ndimage.gaussian_filter(rng.random((n, n)), 1.0)
    big = ndimage.zoom(small, cell, order=1)[:SIZE, :SIZE]
    return ((big - big.min()) / (big.max() - big.min())).astype(np.float32)


def load(prep):
    """Load the prepared grids for one area into module globals."""
    global X0, SIZE, HALF, G, BUILDINGS, BT, NOISE, NOISE2, TREES, TORDER, TKEY_SORTED
    meta = json.load(open(os.path.join(prep, "meta.json")))
    X0, SIZE, HALF = meta["X0"], meta["SIZE"], meta["HALF"]
    G = {k: np.load(os.path.join(prep, f"{k}.npy"), mmap_mode="r") for k in GRIDS}
    BUILDINGS = json.load(open(os.path.join(prep, "buildings.json")))
    BT = building_table()
    NOISE, NOISE2 = smooth_noise(1, 6), smooth_noise(2, 3)
    # trees bucketed by chunk so each chunk only looks at its neighbours
    TREES = np.asarray(G["trees"])
    key = ((TREES[:, 1] + X0) // 16 + 1000) * 4000 + ((TREES[:, 0] + X0) // 16 + 1000)
    TORDER = np.argsort(key, kind="stable")
    TKEY_SORTED = key[TORDER]
    return meta


YS = np.arange(MIN_Y, MIN_Y + HEIGHT)[:, None, None]


def trees_near(cx, cz):
    out = []
    for dz in (-1, 0, 1):
        for dx in (-1, 0, 1):
            k = (cz + dz + 1000) * 4000 + (cx + dx + 1000)
            a, b = np.searchsorted(TKEY_SORTED, [k, k + 1])
            out.append(TORDER[a:b])
    return np.concatenate(out)


def draw_tree(A, x0, z0, tx, tz, base, height, species, blocked):
    """Draw one tree into chunk array A (chunk origin x0, z0). base = first block above ground.
    blocked: (16, 16) columns where no leaves may go (buildings, bridges)."""
    h = int(np.clip(round(height), 3, 30))
    top = base + h - 1
    if species == 0:  # spruce: narrow layered cone
        rmax = np.clip(0.16 * h + 0.6, 1.5, 3.8)
        cbot = base + max(1, int(h * 0.22))
    elif species == 1:  # pine: tall bare trunk, rounded crown at the top
        rmax = np.clip(0.12 * h + 1.0, 1.5, 3.5)
        cbot = base + int(h * 0.62)
    else:  # birch: oval crown on the upper half
        rmax = np.clip(0.14 * h + 0.8, 1.5, 3.2)
        cbot = base + int(h * 0.4)
    R = int(np.ceil(rmax))
    lx0, lx1 = max(tx - R, x0), min(tx + R, x0 + 15)
    lz0, lz1 = max(tz - R, z0), min(tz + R, z0 + 15)
    if lx0 > lx1 or lz0 > lz1:
        return
    xs = np.arange(lx0, lx1 + 1)[None, None, :]
    zs = np.arange(lz0, lz1 + 1)[None, :, None]
    ys = np.arange(cbot, top + 2)[:, None, None]
    d2 = (xs - tx) ** 2 + (zs - tz) ** 2
    jit = hash01(xs + ys * 7, zs - ys * 3, tx * 31 + tz)
    if species == 0:
        frac = (top + 1 - ys) / max(1, top + 1 - cbot)
        r = rmax * frac * np.where((ys - cbot) % 2 == 0, 1.0, 0.72) + 0.35
        leaf = d2 <= r * r - jit
        leaf |= (ys == top + 1) & (d2 == 0)
    else:
        cy = (cbot + top) / 2 + (0.5 if species == 1 else 0)
        rv = max(1.5, (top - cbot) / 2 + 1)
        e = d2 / (rmax * rmax) + ((ys - cy) / rv) ** 2
        leaf = e <= 1.0 - 0.35 * jit
    view = A[cbot - MIN_Y:top + 2 - MIN_Y, lz0 - z0:lz1 - z0 + 1, lx0 - x0:lx1 - x0 + 1]
    leaf &= ~blocked[None, lz0 - z0:lz1 - z0 + 1, lx0 - x0:lx1 - x0 + 1]
    view[leaf & (view == AIR)] = LEAVES[species]
    if x0 <= tx <= x0 + 15 and z0 <= tz <= z0 + 15:
        col = A[base - MIN_Y:top - MIN_Y, tz - z0, tx - x0]
        col[(col == AIR) | np.isin(col, LEAVES_ALL)] = LOG[species]


LEAVES_ALL = np.array(list(LEAVES.values()), np.uint16)
PLANTS = np.array([SHORT_GRASS, FERN, BUSH, FIREFLY, DRY_GRASS, FERN_LO, FERN_HI, TALL_LO, TALL_HI]
                  + list(FLOWERS) + list(LITTER), np.uint16)


def build_chunk(cx, cz):
    x0, z0 = cx * 16, cz * 16
    r0, c0 = z0 - X0, x0 - X0
    w = lambda k: np.asarray(G[k][r0:r0 + 16, c0:c0 + 16])
    w1 = lambda k: np.asarray(G[k][r0 - 1:r0 + 17, c0 - 1:c0 + 17])
    g = w("ground").astype(np.int32)
    wy = w("water_y").astype(np.int32)
    by = w("bottom_y").astype(np.int32)
    lc, road, urban = w("lc"), w("road"), w("urban")
    stream = w("stream") & (road == 0)
    X = np.broadcast_to(x0 + np.arange(16)[None, :], (16, 16))
    Z = np.broadcast_to(z0 + np.arange(16)[:, None], (16, 16))
    rnd, rnd2 = hash01(X, Z, 1), hash01(X, Z, 2)
    n1 = NOISE[r0:r0 + 16, c0:c0 + 16]
    n2 = NOISE2[r0:r0 + 16, c0:c0 + 16]
    water = wy > NONE

    # --- surface and subsurface materials
    surf = np.full((16, 16), GRASS, np.uint16)
    sub = np.full((16, 16), DIRT, np.uint16)
    wild = np.isin(lc, [0, 13, 14]) & ~urban
    surf[wild] = np.where(n1 > 0.64, PODZOL, np.where(n1 < 0.28, MOSS, np.where(rnd < 0.02, COARSE, GRASS)))[wild]
    fid = w("field_id")
    ftype = fid % 3
    rows = np.where(fid % 2 == 0, X, Z) % 2 == 0
    surf[(lc == 1) & (ftype == 0)] = COARSE
    surf[(lc == 1) & (ftype == 1)] = np.where(rows, FARMLAND, COARSE)[(lc == 1) & (ftype == 1)]
    surf[lc == 7] = ASPHALT
    m = lc == 8
    surf[m] = np.where(n2 > 0.62, ANDESITE, np.where(rnd < 0.15, COBBLE, np.where(rnd < 0.25, MOSSY_COBBLE, STONE)))[m]
    m = lc == 9
    surf[m] = np.where(rnd < 0.4, COBBLE, np.where(rnd < 0.6, MOSSY_COBBLE, STONE))[m]
    sub[(lc == 8) | (lc == 9)] = STONE
    m = lc == 10
    surf[m] = np.where(n2 > 0.5, SAND, GRAVEL)[m]
    sub[m] = surf[m]
    surf[lc == 11] = MOSS
    m = lc == 12
    surf[m] = np.where(rnd < 0.55, MOSS, GRASS)[m]
    sub[(lc == 11) | (lc == 12)] = MUD
    m = lc == 15
    surf[m] = np.where(rnd < 0.5, GRAVEL, COARSE)[m]
    puddle = (lc == 11) & (rnd2 < 0.06) & (n2 > 0.45)

    # roads
    road_mat = {1: ASPHALT, 2: ASPHALT, 6: DIRT_PATH, 7: GRAVEL}
    for k, v in road_mat.items():
        surf[road == k] = v
    m = road == 3
    surf[m] = np.where(urban, ASPHALT, np.where(rnd < 0.75, GRAVEL, COARSE))[m]
    m = road == 4
    surf[m] = np.where(rnd < 0.6, COARSE, GRAVEL)[m]
    m = road == 5
    surf[m] = np.where(urban, PAVEMENT, GRAVEL)[m]
    sub[road == 7] = GRAVEL

    # water bottoms and streams
    top = np.where(water, by, g)
    depth = wy - by
    bottom_mat = np.where(depth <= 2, SAND, np.where(n2 > 0.55, GRAVEL, np.where(n2 < 0.3, CLAY, MUD)))
    surf = np.where(water, bottom_mat, surf)
    sub = np.where(water, np.where(depth <= 2, SAND, bottom_mat), sub)
    wtop = np.where(water, wy, NONE)
    s = stream & ~water
    top = np.where(s, g - 2, top)
    surf[s] = GRAVEL
    wtop = np.where(s, g - 1, wtop)

    # --- terrain columns
    A = np.where(YS < top - 3, STONE, np.where(YS < top, sub, np.where(YS == top, surf, AIR))).astype(np.uint16)
    A[(YS < 0) & (A == STONE)] = DEEPSLATE
    A[0] = BEDROCK
    A[(YS > top) & (YS <= wtop)] = WATER
    A[(YS == top) & puddle] = WATER

    # --- bridges
    deck1 = w1("deck_y").astype(np.int32)
    deck = deck1[1:-1, 1:-1]
    bcls = w("bridge_cls")
    has_deck = deck > NONE
    raised = has_deck & ((deck >= top + 2) | water)
    low = has_deck & ~raised  # bridge over a ditch: just road at ground level
    A[(YS == top) & low & (bcls <= 3)] = ASPHALT
    A[(YS == top) & low & (bcls > 3)] = PAVEMENT
    if raised.any():
        wood = np.isin(bcls, [4, 6])
        deck_mat = np.where(bcls <= 3, ASPHALT, np.where(bcls == 7, GRAVEL, np.where(wood, PLANKS_DECK, PAVEMENT)))
        r1 = deck1 > NONE
        edge = raised & ~(r1[:-2, 1:-1] & r1[2:, 1:-1] & r1[1:-1, :-2] & r1[1:-1, 2:])
        A[(YS == deck) & raised] = np.broadcast_to(deck_mat, A.shape)[(YS == deck) & raised]
        A[(YS == deck - 1) & raised & ~wood & (deck - 1 > top)] = FOUNDATION
        A[(YS == deck + 1) & edge] = np.broadcast_to(np.where(wood, SLAB_WOOD, SLAB_STONE), A.shape)[(YS == deck + 1) & edge]
        A[(YS > deck + 1) & (YS <= deck + 4) & raised] = AIR
        A[(YS > top) & (YS < deck) & raised & w("pillar")] = FOUNDATION

    # --- buildings
    bid1 = w1("bid")
    bid = bid1[1:-1, 1:-1]
    inb = bid > 0
    if inb.any():
        bd1 = w1("bdist").astype(np.int32)
        bd = bd1[1:-1, 1:-1]
        yf, hw, flat, cap = BT["floor"][bid], BT["hw"][bid], BT["flat"][bid].astype(bool), BT["cap"][bid]
        wall = inb & (bd == 1)
        interior = inb & ~wall
        A[(YS > yf) & inb] = AIR
        A[(YS > top) & (YS < yf) & inb] = FOUNDATION
        fl = (YS == yf)
        A[fl & interior] = np.broadcast_to(BT["floorblk"][bid], A.shape)[fl & interior]
        A[fl & wall] = FOUNDATION
        walls = (YS > yf) & (YS <= yf + hw) & wall
        A[walls] = np.broadcast_to(BT["wall"][bid], A.shape)[walls]
        u = (X + Z)
        winmod = BT["winmod"][bid]
        win = walls & ((YS - yf) % 3 == 2) & np.where(winmod == 4, u % 4 == 0, u % 3 != 0)
        A[win] = GLASS
        storeys = (YS > yf) & (YS < yf + hw) & ((YS - yf) % 3 == 0) & interior
        A[storeys] = np.broadcast_to(BT["floorblk"][bid], A.shape)[storeys]
        # flat roofs with a low parapet
        fr = (YS == yf + hw + 1) & interior & flat
        A[fr] = np.broadcast_to(BT["flatroof"][bid], A.shape)[fr]
        par = (YS > yf + hw) & (YS <= yf + hw + 2) & wall & flat
        A[par] = np.broadcast_to(BT["wall"][bid], A.shape)[par]
        # hip roofs from the distance to the footprint edge, with stairs on the slopes
        hip = inb & ~flat
        if hip.any():
            rd = np.minimum(bd, cap)
            same = lambda dz, dx: (bid1[1 + dz:17 + dz, 1 + dx:17 + dx] == bid) & (bd1[1 + dz:17 + dz, 1 + dx:17 + dx] == bd + 1)
            up = np.stack([same(-1, 0), same(1, 0), same(0, -1), same(0, 1)])  # N S W E
            nup = up.sum(axis=0)
            facing = up.argmax(axis=0)
            roofblk = np.where((nup == 1) & (bd < cap), ROOF_STAIRS[BT["roof"][bid], facing], ROOF_FULL[BT["roof"][bid]])
            ry = yf + hw + rd
            r = (YS == ry) & hip
            A[r] = np.broadcast_to(roofblk, A.shape)[r]
            ceil = (YS == yf + hw + 1) & hip & interior & (rd > 1)
            A[ceil] = np.broadcast_to(BT["floorblk"][bid], A.shape)[ceil]

    # --- trees
    blocked = inb | has_deck
    for i in trees_near(cx, cz):
        col, row, h10, sp = TREES[i]
        if G["stream"][row, col] or G["deck_y"][row, col] > NONE:
            continue
        gy = int(G["ground"][row, col])
        draw_tree(A, x0, z0, col + X0, row + X0, gy + 1, h10 / 10, sp, blocked)

    # --- ground plants
    above = np.take_along_axis(A, (top + 1 - MIN_Y)[None].clip(0, HEIGHT - 1), axis=0)[0]
    above2 = np.take_along_axis(A, (top + 2 - MIN_Y)[None].clip(0, HEIGHT - 1), axis=0)[0]
    base = np.take_along_axis(A, (top - MIN_Y)[None].clip(0, HEIGHT - 1), axis=0)[0]
    free = (above == AIR) & np.isin(base, [GRASS, PODZOL, MOSS, COARSE]) & ~water & ~s & (road == 0) & ~inb & ~has_deck
    plant = np.zeros((16, 16), np.uint16)
    tall_upper = np.zeros((16, 16), np.uint16)
    p = rnd2
    flower = FLOWERS[(hash01(X, Z, 3) * len(FLOWERS)).astype(int)]
    litter = LITTER[(hash01(X, Z, 4) * len(LITTER)).astype(int)]
    forest = wild & (lc != 1)
    plant = np.where(forest & (p < 0.07), SHORT_GRASS, plant)
    plant = np.where(forest & (p >= 0.07) & (p < 0.12), FERN, plant)
    plant = np.where(forest & (p >= 0.12) & (p < 0.135), BUSH, plant)
    plant = np.where(forest & (p >= 0.135) & (p < 0.21), litter, plant)
    big_fern = forest & (p >= 0.21) & (p < 0.22) & (above2 == AIR)
    plant = np.where(big_fern, FERN_LO, plant)
    tall_upper = np.where(big_fern, FERN_HI, tall_upper)
    plant = np.where(forest & (p >= 0.22) & (p < 0.223), FIREFLY, plant)
    yard = ~wild & np.isin(lc, [0, 2, 4, 5, 6, 13, 14])
    plant = np.where(yard & (p < 0.06), SHORT_GRASS, plant)
    plant = np.where(yard & (p >= 0.06) & (p < 0.066), flower, plant)
    meadow = lc == 3
    plant = np.where(meadow & (p < 0.35), SHORT_GRASS, plant)
    tall = meadow & (p >= 0.35) & (p < 0.40) & (above2 == AIR)
    plant = np.where(tall, TALL_LO, plant)
    tall_upper = np.where(tall, TALL_HI, tall_upper)
    plant = np.where(meadow & (p >= 0.40) & (p < 0.43), flower, plant)
    plant = np.where(meadow & (p >= 0.43) & (p < 0.47), DRY_GRASS, plant)
    bog = (lc == 11) | (lc == 12)
    plant = np.where(bog & (p < 0.08), SHORT_GRASS, plant)
    plant = np.where(bog & (p >= 0.08) & (p < 0.12), BUSH, plant)
    plant = np.where(bog & (p >= 0.12) & (p < 0.16), DRY_GRASS, plant)
    field = lc == 1
    plant = np.where(field & (ftype == 0) & (p < 0.5), DRY_GRASS, plant)
    plant = np.where(field & (ftype == 2) & (p < 0.4), SHORT_GRASS, plant)
    put = free & (plant != AIR) & (base != COARSE) | (free & (plant == DRY_GRASS))
    zz, xx = np.nonzero(put)
    yy = top[zz, xx] + 1 - MIN_Y
    A[yy, zz, xx] = plant[zz, xx]
    up = put & (tall_upper != AIR)
    zz, xx = np.nonzero(up)
    A[top[zz, xx] + 2 - MIN_Y, zz, xx] = tall_upper[zz, xx]
    return A


def region_job(args):
    rx, rz, outdir, chunk_range = args
    cmin, cmax = chunk_range
    chunks = {}
    for lz in range(32):
        for lx in range(32):
            cx, cz = rx * 32 + lx, rz * 32 + lz
            if not (cmin[0] <= cx <= cmax[0] and cmin[1] <= cz <= cmax[1]):
                continue
            chunks[(lx, lz)] = encode_chunk(B, cx, cz, build_chunk(cx, cz))
    if chunks:
        write_region(os.path.join(outdir, f"r.{rx}.{rz}.mca"), chunks)
    return len(chunks)


def run(prep, world, jobs=None, progress=print):
    """Write all region files for the area. Returns the number of chunks."""
    load(prep)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    outdir = os.path.join(world, "dimensions", "minecraft", "overworld", "region")
    os.makedirs(outdir, exist_ok=True)
    # whole chunks covering -HALF..HALF that stay inside the grid (with 1 block to spare for neighbours)
    lo, hi = -((-X0 - 1) // 16), (-X0 - 17) // 16
    assert lo * 16 <= -HALF and hi * 16 + 15 >= HALF - 1
    cmin, cmax = (lo, lo), (hi, hi)
    regions = [(rx, rz) for rz in range(lo // 32, hi // 32 + 1) for rx in range(lo // 32, hi // 32 + 1)]
    total_chunks = (hi - lo + 1) ** 2
    t0 = time.time()
    done = 0
    with mp.get_context("fork").Pool(jobs) as pool:
        for n in pool.imap_unordered(region_job, [(rx, rz, outdir, (cmin, cmax)) for rx, rz in regions]):
            done += n
            el = time.time() - t0
            eta = el / max(done, 1) * (total_chunks - done)
            progress(f"  chunks {done}/{total_chunks}  ({el / 60:.1f} min, about {eta / 60:.0f} min left)")
    return done
