import io
import os
import zlib

import nbtlib
import numpy as np
import pytest

from finland2mc import mcformat

BLOCKS = os.path.join(os.path.dirname(mcformat.__file__), "data", "blocks_26.3.json")


def unpack(longs, bits):
    per = 64 // bits
    a = np.asarray(longs, np.int64).view(np.uint64)
    return ((a[:, None] >> (np.arange(per, dtype=np.uint64) * np.uint64(bits))) & np.uint64((1 << bits) - 1)).ravel()[:4096]


@pytest.mark.parametrize("bits", [4, 5, 7, 9])
def test_pack_round_trip(bits):
    idx = np.random.default_rng(bits).integers(0, 1 << bits, 4096)
    assert (unpack(mcformat.pack_indices(idx, bits), bits) == idx).all()


def test_palette_entries_match_the_game():
    # default states are a bare (wrapped) string; anything else lists every property
    b = mcformat.Blocks(BLOCKS)
    stone = nbtlib.Compound.parse(io.BytesIO(b.palette_entry[b("stone")]))
    assert stone == {"": "minecraft:stone"}
    leaves = nbtlib.Compound.parse(io.BytesIO(b.palette_entry[b("oak_leaves[persistent=true,distance=1]")]))
    assert leaves["id"] == "minecraft:oak_leaves"
    assert dict(leaves["properties"]) == {"distance": "1", "persistent": "true", "waterlogged": "false"}


def test_unknown_blocks_are_rejected():
    b = mcformat.Blocks(BLOCKS)
    with pytest.raises(KeyError):
        b("no_such_block")
    with pytest.raises(KeyError):
        b("spruce_log[axis=w]")


def test_chunk_and_region(tmp_path):
    b = mcformat.Blocks(BLOCKS)
    column = np.zeros((mcformat.HEIGHT, 16, 16), np.uint16)
    column[:70] = b("stone")
    column[70, 3, 5] = b("spruce_log[axis=x]")
    raw = mcformat.encode_chunk(b, 2, -3, column)
    path = tmp_path / "r.0.-1.mca"
    mcformat.write_region(path, {(2, 29): raw})
    data = path.read_bytes()
    i = 2 + 29 * 32
    off = int.from_bytes(data[i * 4:i * 4 + 3], "big") * 4096
    length = int.from_bytes(data[off:off + 4], "big")
    chunk = nbtlib.File.parse(io.BytesIO(zlib.decompress(data[off + 5:off + 4 + length])))
    assert (chunk["xPos"], chunk["zPos"], chunk["DataVersion"]) == (2, -3, mcformat.DATA_VERSION)
    sec = next(s for s in chunk["sections"] if s["Y"] == 0)  # y = 0..15, the section holding y = 6
    palette = [p.get("id", p.get("")) for p in sec["block_states"]["palette"]]
    idx = unpack(sec["block_states"]["data"], 4).reshape(16, 16, 16)
    assert palette[idx[6, 3, 5]] == "minecraft:spruce_log"
    assert palette[idx[6, 0, 0]] == "minecraft:air"
    assert palette[idx[5, 0, 0]] == "minecraft:stone"
