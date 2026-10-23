"""Top-down map of a generated world, read back from its region files."""
import glob
import io
import multiprocessing as mp
import os
import re
import struct
import zlib

import nbtlib
import numpy as np
from PIL import Image

COLORS = {
    "grass_block": (98, 140, 72), "podzol": (106, 82, 48), "moss_block": (90, 110, 50), "coarse_dirt": (119, 85, 59),
    "dirt": (134, 96, 67), "farmland": (100, 70, 45), "stone": (125, 125, 125), "andesite": (136, 136, 137),
    "cobblestone": (120, 120, 120), "mossy_cobblestone": (100, 115, 90), "sand": (219, 207, 163), "gravel": (131, 127, 126),
    "clay": (160, 166, 179), "mud": (60, 57, 60), "water": (52, 88, 180), "gray_concrete": (70, 72, 76),
    "polished_andesite": (150, 152, 152), "dirt_path": (148, 122, 65), "spruce_leaves": (55, 90, 55),
    "oak_leaves": (75, 120, 60), "birch_leaves": (110, 150, 75), "spruce_log": (60, 45, 30), "birch_log": (200, 200, 190),
    "short_grass": (98, 150, 72), "fern": (80, 130, 60), "bush": (70, 120, 50), "leaf_litter": (150, 110, 60),
    "short_dry_grass": (190, 170, 100), "tall_grass": (98, 150, 72), "large_fern": (80, 130, 60),
    "stone_bricks": (122, 121, 122), "stone_brick_slab": (122, 121, 122), "spruce_planks": (115, 85, 49),
    "spruce_slab": (115, 85, 49), "glass": (200, 220, 230), "dandelion": (98, 150, 72), "poppy": (98, 150, 72),
    "oxeye_daisy": (98, 150, 72), "cornflower": (98, 150, 72), "firefly_bush": (70, 120, 50), "bedrock": (60, 60, 60),
}
ROOF_LIKE = (150, 60, 50)


def color_of(name):
    n = name.split("[")[0].replace("minecraft:", "")
    if n in COLORS:
        return COLORS[n]
    if "stairs" in n or "tiles" in n or "bricks" in n or "blackstone" in n or "planks" in n:
        return ROOF_LIKE
    if "concrete" in n or "terracotta" in n:
        return (190, 180, 170)
    return (200, 0, 200)


def region_tops(path):
    """Top non-air block name and height for every column in a region file."""
    data = open(path, "rb").read()
    rx, rz = map(int, re.findall(r"r\.(-?\d+)\.(-?\d+)\.mca", path)[0])
    names = {}
    top_id = np.zeros((512, 512), np.int32)
    top_y = np.full((512, 512), -999, np.int32)
    for i in range(1024):
        off = int.from_bytes(data[i * 4:i * 4 + 3], "big")
        if not off:
            continue
        o = off * 4096
        ln = struct.unpack(">I", data[o:o + 4])[0]
        chunk = nbtlib.File.parse(io.BytesIO(zlib.decompress(data[o + 5:o + 4 + ln])))
        lx, lz = i % 32, i // 32
        found = np.zeros((16, 16), bool)
        tid = np.zeros((16, 16), np.int32)
        ty = np.full((16, 16), -999, np.int32)
        for s in sorted(chunk["sections"], key=lambda s: -int(s["Y"])):
            if "block_states" not in s:
                continue
            pal = s["block_states"]["palette"]
            pal = [str(e) if isinstance(e, str) else str(e.get("", e.get("id"))) for e in pal]
            if len(pal) == 1:
                if pal[0] == "minecraft:air":
                    continue
                idx = np.zeros(4096, np.int64)
            else:
                bits = max(4, int(np.ceil(np.log2(len(pal)))))
                per = 64 // bits
                a = np.asarray(s["block_states"]["data"], np.int64).view(np.uint64)
                idx = ((a[:, None] >> (np.arange(per, dtype=np.uint64) * np.uint64(bits))) & np.uint64((1 << bits) - 1)).ravel()[:4096].astype(np.int64)
            ids = np.array([names.setdefault(p, len(names)) for p in pal])[idx].reshape(16, 16, 16)
            air = names.setdefault("minecraft:air", len(names))
            for y in range(15, -1, -1):
                layer = ids[y]
                new = (layer != air) & ~found
                tid[new] = layer[new]
                ty[new] = int(s["Y"]) * 16 + y
                found |= new
            if found.all():
                break
        top_id[lz * 16:lz * 16 + 16, lx * 16:lx * 16 + 16] = tid
        top_y[lz * 16:lz * 16 + 16, lx * 16:lx * 16 + 16] = ty
    inv = {v: k for k, v in names.items()}
    pal_rgb = np.array([color_of(inv.get(i, "air")) for i in range(max(len(names), 1))], np.uint8)
    return rx, rz, pal_rgb[top_id], top_y


def render(world, out, half, max_px=2000):
    """Save a top-down map of the playable area, at most max_px wide."""
    files = glob.glob(os.path.join(world, "dimensions/minecraft/overworld/region/*.mca"))
    with mp.get_context("fork").Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(region_tops, files)
    rxs, rzs = [r[0] for r in res], [r[1] for r in res]
    ox, oz = min(rxs) * 512, min(rzs) * 512
    W, H = (max(rxs) - min(rxs) + 1) * 512, (max(rzs) - min(rzs) + 1) * 512
    rgb = np.zeros((H, W, 3), np.uint8)
    hy = np.full((H, W), -999, np.int32)
    for rx, rz, c, y in res:
        sl = np.s_[rz * 512 - oz:rz * 512 - oz + 512, rx * 512 - ox:rx * 512 - ox + 512]
        rgb[sl] = c
        hy[sl] = y
    sl = np.s_[-half - oz:half - oz, -half - ox:half - ox]
    rgb, hy = rgb[sl], hy[sl]
    # light from the north-west
    shade = np.clip(1 + 0.12 * ((np.roll(hy, 1, 0) - hy) + (np.roll(hy, 1, 1) - hy)), 0.55, 1.35)
    im = Image.fromarray(np.clip(rgb * shade[..., None], 0, 255).astype(np.uint8))
    if im.width > max_px:
        im = im.resize((max_px, max_px), Image.LANCZOS)
    im.save(out)
