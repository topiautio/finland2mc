"""Find and download National Land Survey of Finland open data tiles (Funet mirror), with a local cache.

Map sheets follow the TM35 sheet division: 1:200 000 sheets are 192 x 96 km (rows K..X, columns 2..6),
split 2 x 2 down to 24 x 12 km sheets (1 = SW, 2 = NW, 3 = SE, 4 = NE). Those split into eight 6 x 6 km
tiles A..H (4 columns x 2 rows, A = west bottom, B = west top, ...). The topographic database is
published as 12 x 12 km halves (L = west, R = east) of the 24 x 12 km sheets.
"""
import os
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyogrio.raw as pr
import shapely

from .geocode import USER_AGENT

BASE = "https://www.nic.funet.fi/index/geodata/mml"
CACHE = Path(os.environ.get("FINLAND2MC_CACHE", Path.home() / ".cache" / "finland2mc"))
MTK_YEAR = "2025"
ROWS = "KLMNPQRSTUVWX"
E0, N0 = -76000, 6570000  # SW corner of sheet K2


def sheet_24k(e, n):
    """Name and bounds (w, s, e, n) of the 24 x 12 km map sheet containing the point."""
    c = int((e - E0) // 192000)
    r = int((n - N0) // 96000)
    if not (0 <= r < len(ROWS) and 0 <= c <= 4):
        raise ValueError("outside the Finnish map sheet grid")
    name = f"{ROWS[r]}{c + 2}"
    w, s, width, height = E0 + c * 192000, N0 + r * 96000, 192000, 96000
    for _ in range(3):
        width, height = width / 2, height / 2
        col, row = int((e - w) // width), int((n - s) // height)
        name += "1234"[col * 2 + row]
        w, s = w + col * width, s + row * height
    return name, (w, s, w + width, s + height)


def tile_6k(e, n):
    name, (w, s, _, _) = sheet_24k(e, n)
    col, row = int((e - w) // 6000), int((n - s) // 6000)
    return name + "ABCDEFGH"[col * 2 + row]


def mtk_sheet(e, n):
    name, (w, *_) = sheet_24k(e, n)
    return name + ("L" if e < w + 12000 else "R")


def _grid_points(bounds, step):
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0, x1 + step, step).clip(x0, x1 - 1)
    ys = np.arange(y0, y1 + step, step).clip(y0, y1 - 1)
    return [(x, y) for x in xs for y in ys]


def _sub(name):
    return f"{name[:2]}/{name[:3]}"


def dem_paths(bounds):
    names = sorted({tile_6k(x, y) for x, y in _grid_points(bounds, 3000)})
    return [f"dem2m/2008_latest/{_sub(t)}/{t}.tif" for t in names]


def mtk_paths(bounds):
    names = sorted({mtk_sheet(x, y) for x, y in _grid_points(bounds, 6000)})
    return [f"maastotietokanta/{MTK_YEAR}/shp/{_sub(s)}/{s}.shp.zip" for s in names]


def lidar_paths(bounds):
    """Newest laser scanning tile for every 3 x 3 km square in the area (via the national index)."""
    index = [fetch(f"laserkeilaus/2008_latest/2008_latest.{ext}") for ext in ("shp", "shx", "dbf", "prj")][0]
    meta, _, geom, fields = pr.read(index, columns=["path"], bbox=tuple(bounds))
    if geom is None:
        return []
    out = []
    for g, path in zip(geom, fields[0]):
        if shapely.from_wkb(g).intersects(shapely.box(*bounds)):
            out.append(str(path).split("mml/", 1)[1])  # 'mml/laserkeilaus/...laz'
    return sorted(set(out))


def local(rel):
    return CACHE / "nls" / rel


def remote_size(rel):
    """Size in bytes, or None if the file does not exist (e.g. a sea-only tile)."""
    req = urllib.request.Request(f"{BASE}/{rel}", method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def fetch(rel):
    """Download a file into the cache (once). Returns its local path, or None if it doesn't exist."""
    dest = local(rel)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
    return None


def plan(bounds, lidar=True):
    """Files needed for an area: {kind: [relative paths]} plus total bytes still to download."""
    files = dict(dem=dem_paths(bounds), mtk=mtk_paths(bounds), lidar=lidar_paths(bounds) if lidar else [])
    todo = [p for ps in files.values() for p in ps if not local(p).exists()]
    with ThreadPoolExecutor(8) as ex:
        sizes = dict(zip(todo, ex.map(remote_size, todo)))
    missing = {p for p, s in sizes.items() if s is None}
    files = {k: [p for p in v if p not in missing] for k, v in files.items()}
    return files, sum(s for s in sizes.values() if s)


def download(files, progress=print):
    paths = [p for ps in files.values() for p in ps]
    done = 0
    with ThreadPoolExecutor(6) as ex:
        for _ in ex.map(fetch, paths):
            done += 1
            progress(f"  downloaded {done}/{len(paths)}")
    out = {}
    for kind, ps in files.items():
        out[kind] = [local(p) for p in ps if local(p).exists()]
    # unpack topographic database sheets next to their zip files
    dirs = []
    for z in out["mtk"]:
        d = z.with_name(z.name.replace(".shp.zip", ""))
        if not d.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
        dirs.append(d)
    out["mtk"] = dirs
    return out
