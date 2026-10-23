"""Add level.dat and the other world files to the generated region files.

The template in data/template/ is an empty world saved by the official 26.3 server.
"""
import json
import os
import shutil
import time

import nbtlib
import numpy as np
from nbtlib.tag import Byte, Double, Int, IntArray, Long, String

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def pick_spawn(prep):
    """Nearest paved road to the middle of the area with no tree right next to it."""
    meta = json.load(open(os.path.join(prep, "meta.json")))
    road = np.load(os.path.join(prep, "road.npy"), mmap_mode="r")
    ground = np.load(os.path.join(prep, "ground.npy"), mmap_mode="r")
    water = np.load(os.path.join(prep, "water_y.npy"), mmap_mode="r")
    trees = np.load(os.path.join(prep, "trees.npy"))
    mid = -meta["X0"]
    win = min(150, meta["HALF"] - 1)
    rr, cc = np.mgrid[mid - win:mid + win, mid - win:mid + win]
    ok = np.isin(road[rr, cc], [1, 2, 3, 5])
    near = trees[(np.abs(trees[:, 0] - mid) < win + 4) & (np.abs(trees[:, 1] - mid) < win + 4)]
    for tx, tz in near[:, :2]:
        ok &= (rr - tz) ** 2 + (cc - tx) ** 2 > 16
    if not ok.any():
        ok = water[rr, cc] == -9999
    d = np.where(ok, (rr - mid) ** 2 + (cc - mid) ** 2, 1 << 40)
    i = np.unravel_index(d.argmin(), d.shape)
    row, col = int(rr[i]), int(cc[i])
    return col + meta["X0"], int(ground[row, col]) + 1, row + meta["X0"]


def finalize(world, prep, name):
    meta = json.load(open(os.path.join(prep, "meta.json")))
    shutil.copytree(os.path.join(DATA, "template"), world, dirs_exist_ok=True)
    x, y, z = pick_spawn(prep)
    level = nbtlib.load(os.path.join(world, "level.dat"))
    d = level["Data"]
    d["LevelName"] = String(name)
    d["GameType"] = Int(1)  # creative
    d["allowCommands"] = Byte(1)
    d["LastPlayed"] = Long(int(time.time() * 1000))
    d["difficulty_settings"]["difficulty"] = String("peaceful")
    d["spawn"]["pos"] = IntArray([x, y, z])
    level.save()
    border = nbtlib.load(os.path.join(world, "dimensions/minecraft/overworld/data/minecraft/world_border.dat"))
    for k in ("size", "lerp_target"):
        border["data"][k] = Double(2.0 * meta["HALF"])
    border["data"]["center_x"] = Double(0.0)
    border["data"]["center_z"] = Double(0.0)
    border.save()
    return x, y, z
