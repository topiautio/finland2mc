"""Install a world into a Prism Launcher instance, creating the instance if needed."""
import configparser
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

from . import MINECRAFT_VERSION

JAVA_KEYS = ("AutomaticJava", "JavaArchitecture", "JavaPath", "JavaRealArchitecture", "JavaSignature",
             "JavaVendor", "JavaVersion", "OverrideJavaLocation")


def find_root():
    home = Path.home()
    candidates = [home / ".local/share/PrismLauncher",
                  home / ".var/app/org.prismlauncher.PrismLauncher/data/PrismLauncher",
                  home / "Library/Application Support/PrismLauncher"]
    if os.environ.get("APPDATA"):
        candidates.append(Path(os.environ["APPDATA"]) / "PrismLauncher")
    for c in candidates:
        if (c / "prismlauncher.cfg").exists():
            return c
    return None


def _read_cfg(path):
    cp = configparser.ConfigParser(interpolation=None, strict=False, delimiters=("=",))
    cp.optionxform = str
    text = path.read_text(errors="replace")
    if not text.lstrip().startswith("["):
        text = "[General]\n" + text
    cp.read_string(text)
    return cp["General"] if "General" in cp else {}


def instances_dir(root):
    d = _read_cfg(root / "prismlauncher.cfg").get("InstanceDir", "instances")
    return Path(d) if os.path.isabs(d) else root / d


def instances(root):
    """{display name: (folder, minecraft version)}"""
    out = {}
    base = instances_dir(root)
    if not base.exists():
        return out
    for d in base.iterdir():
        cfg = d / "instance.cfg"
        if not cfg.exists():
            continue
        name = _read_cfg(cfg).get("name", d.name)
        version = None
        try:
            pack = json.load(open(d / "mmc-pack.json"))
            version = next((c.get("version") for c in pack["components"] if c.get("uid") == "net.minecraft"), None)
        except (OSError, ValueError, KeyError):
            pass
        out[name] = (d, version)
    return out


def prism_running():
    if not sys.platform.startswith("linux"):
        return False
    for p in Path("/proc").glob("[0-9]*/comm"):
        try:
            if "prismlauncher" in p.read_text().lower():
                return True
        except OSError:
            pass
    return False


def _create_instance(root, name):
    base = instances_dir(root)
    folder = re.sub(r"[^\w\- ]", "", name.replace(":", "-")).strip() or "finland2mc"
    while (base / folder).exists():
        folder += "_"
    inst = base / folder
    (inst / "minecraft" / "saves").mkdir(parents=True)
    # reuse the setup of an instance that already runs this Minecraft version, if there is one
    donor = next((d for d, v in instances(root).values() if v == MINECRAFT_VERSION and d != inst), None)
    lines = ["[General]", "ConfigVersion=1.3", "InstanceType=OneSix"]
    if donor:
        shutil.copy(donor / "mmc-pack.json", inst / "mmc-pack.json")
        cfg = _read_cfg(donor / "instance.cfg")
        lines += [f"{k}={cfg[k]}" for k in JAVA_KEYS if k in cfg]
    else:
        components = []
        meta = root / "meta" / "net.minecraft" / f"{MINECRAFT_VERSION}.json"
        if meta.exists():
            for req in json.load(open(meta)).get("requires", []):
                v = req.get("suggests") or req.get("equals")
                if v:
                    components.append(dict(uid=req["uid"], version=v, dependencyOnly=True))
        components.append(dict(uid="net.minecraft", version=MINECRAFT_VERSION, important=True))
        json.dump(dict(components=components, formatVersion=1), open(inst / "mmc-pack.json", "w"), indent=4)
    lines += ["iconKey=grass", f"name={name}", f"uuid={uuid.uuid4().hex}"]
    (inst / "instance.cfg").write_text("\n".join(lines) + "\n")
    return inst


def install(world, instance_name):
    """Copy the world folder into the instance's saves. Returns (instance folder, world path, created)."""
    root = find_root()
    if root is None:
        raise RuntimeError("Prism Launcher isn't installed (no prismlauncher.cfg found)")
    found = instances(root).get(instance_name)
    created = found is None
    inst = _create_instance(root, instance_name) if created else found[0]
    mc = inst / ".minecraft" if (inst / ".minecraft").exists() else inst / "minecraft"
    saves = mc / "saves"
    saves.mkdir(parents=True, exist_ok=True)
    dest = saves / Path(world).name
    n = 2
    while dest.exists():
        dest = saves / f"{Path(world).name} ({n})"
        n += 1
    shutil.copytree(world, dest)
    return inst, dest, created
