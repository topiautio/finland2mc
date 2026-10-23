"""Minimal writer for Minecraft Java Edition 26.3 chunks and region files.

Only what this project needs: a block registry checked against the game's own block report,
a small NBT encoder, section packing and the .mca region container.
"""
import json
import re
import struct
import time
import zlib

import numpy as np

DATA_VERSION = 5023  # Minecraft 26.3
MIN_Y, HEIGHT = -64, 384
N_SECTIONS = HEIGHT // 16


# --- NBT encoding -------------------------------------------------------------------------
def _s(text):
    b = text.encode("utf-8")
    return struct.pack(">H", len(b)) + b


def _tag(kind, name, payload):
    return bytes([kind]) + _s(name) + payload


def t_byte(name, v):
    return _tag(1, name, struct.pack(">b", v))


def t_int(name, v):
    return _tag(3, name, struct.pack(">i", v))


def t_long(name, v):
    return _tag(4, name, struct.pack(">q", v))


def t_string(name, v):
    return _tag(8, name, _s(v))


def t_long_array(name, arr):
    return _tag(12, name, struct.pack(">i", len(arr)) + np.asarray(arr, ">i8").tobytes())


def t_list(name, elem_kind, payloads):
    payloads = list(payloads)
    return _tag(9, name, bytes([elem_kind if payloads else 0]) + struct.pack(">i", len(payloads)) + b"".join(payloads))


def t_compound(name, *tags):
    return _tag(10, name, b"".join(tags) + b"\x00")


def compound_payload(*tags):
    return b"".join(tags) + b"\x00"


# --- Block registry -----------------------------------------------------------------------
class Blocks:
    """Global block-state ids. Specs look like 'spruce_log[axis=x]'; unknown names or values raise."""

    SPEC = re.compile(r"^(?:minecraft:)?([a-z0-9_]+)(?:\[(.*)\])?$")

    def __init__(self, report_path):
        self.report = json.load(open(report_path))
        self.ids = {}
        self.palette_entry = []  # encoded palette compound payload per id
        self.names = []
        self("air")

    def __call__(self, spec):
        if spec in self.ids:
            return self.ids[spec]
        m = self.SPEC.match(spec)
        if not m:
            raise ValueError(f"bad block spec {spec!r}")
        name = "minecraft:" + m.group(1)
        info = self.report.get(name)
        if info is None:
            raise KeyError(f"block {name} does not exist in 26.3")
        props = dict(kv.split("=") for kv in m.group(2).split(",")) if m.group(2) else {}
        allowed = info.get("properties", {})
        for k, v in props.items():
            if k not in allowed or v not in allowed[k]:
                raise KeyError(f"{name} has no property {k}={v}")
        default = next(s.get("properties", {}) for s in info["states"] if s.get("default"))
        full = {**default, **props}
        if full == default:
            # default states are stored as a bare string, wrapped like the game does in mixed lists
            entry = compound_payload(t_string("", name))
        else:
            entry = compound_payload(t_string("id", name),
                                     t_compound("properties", *[t_string(k, v) for k, v in full.items()]))
        bid = len(self.palette_entry)
        self.ids[spec] = bid
        self.palette_entry.append(entry)
        self.names.append(spec)
        return bid


# --- Chunks and regions -------------------------------------------------------------------
def pack_indices(idx, bits):
    per = 64 // bits
    n = -(-4096 // per)
    a = np.zeros(n * per, np.uint64)
    a[:4096] = idx
    a = a.reshape(n, per) << (np.arange(per, dtype=np.uint64) * np.uint64(bits))
    return np.bitwise_or.reduce(a, axis=1).view(np.int64)


def encode_section(blocks, y, sec, biome_payload):
    u, inv = np.unique(sec, return_inverse=True)
    tags = [t_list("palette", 10, [blocks.palette_entry[i] for i in u])]
    if len(u) > 1:
        bits = max(4, int(np.ceil(np.log2(len(u)))))
        tags.append(t_long_array("data", pack_indices(inv.ravel(), bits)))
    return compound_payload(t_byte("Y", y), t_compound("block_states", *tags), biome_payload)


# Not taiga: superflat worlds have sea level -63, so the game treats everything here as high
# altitude and taiga lakes freeze over. Birch forest stays above freezing at every height used.
def encode_chunk(blocks, cx, cz, column, biome="minecraft:birch_forest"):
    """column: uint16 array (384, 16, 16) indexed [y - MIN_Y, z, x]. Returns uncompressed NBT bytes."""
    biome_payload = t_compound("biomes", t_list("palette", 8, [_s(biome)]))
    sections = [encode_section(blocks, s + MIN_Y // 16, column[s * 16:(s + 1) * 16], biome_payload)
                for s in range(N_SECTIONS)]
    empty_list = b"\x00" + struct.pack(">i", 0)
    root = compound_payload(
        t_int("DataVersion", DATA_VERSION),
        t_int("xPos", cx), t_int("zPos", cz), t_int("yPos", MIN_Y // 16),
        t_string("Status", "minecraft:full"),
        t_long("LastUpdate", 0), t_long("InhabitedTime", 0),
        t_list("sections", 10, sections),
        t_list("block_entities", 10, []),
        t_list("block_ticks", 10, []),
        t_list("fluid_ticks", 10, []),
        t_list("PostProcessing", 9, [empty_list] * N_SECTIONS),
        t_compound("structures", t_compound("References"), t_compound("starts")),
        t_byte("isLightOn", 0),  # let the game compute lighting on first load
    )
    return b"\x0a" + _s("") + root


def write_region(path, chunks):
    """chunks: {(local_x, local_z): uncompressed chunk NBT bytes}"""
    header = bytearray(8192)
    body = bytearray()
    sector = 2
    now = int(time.time())
    for (lx, lz), raw in sorted(chunks.items()):
        comp = zlib.compress(raw, 6)
        data = struct.pack(">I", len(comp) + 1) + b"\x02" + comp
        data += b"\x00" * ((-len(data)) % 4096)
        n = len(data) // 4096
        assert n < 256, "chunk too large for region sector table"
        i = lx + lz * 32
        header[i * 4:i * 4 + 4] = ((sector << 8) | n).to_bytes(4, "big")
        header[4096 + i * 4:4096 + i * 4 + 4] = now.to_bytes(4, "big")
        body += data
        sector += n
    with open(path, "wb") as f:
        f.write(header)
        f.write(body)
