"""finland2mc: give it a place in Finland, get a 1:1 Minecraft world of it.

    finland2mc "Jyväskylä"                 # asks a few questions, everything has a default
    finland2mc "Porvoo" --size 2 --yes     # no questions
    finland2mc --at 62.2416,25.7496        # exact centre point (lat,lon or E...,N... in ETRS-TM35FIN)
"""
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

from . import MINECRAFT_VERSION, __version__, geocode, nls, prism
from .area import Area

SECONDS_PER_CHUNK = (0.025, 0.12)  # per CPU core: town centre, dense forest
DEFAULT_INSTANCE = "Finland 1:1"


def say(*a):
    print(*a, flush=True)


def ask(question, default, interactive):
    if not interactive:
        return default
    ans = input(f"{question} [{default}]: ").strip()
    return ans or default


def ask_yes(question, default, interactive):
    if not interactive:
        return default
    while True:
        ans = input(f"{question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes", "k", "kyllä"):
            return True
        if ans in ("n", "no", "e", "ei"):
            return False
        say("Please answer y or n.")


def choose_place(args, interactive):
    if args.at:
        return geocode.from_coordinates(args.at)
    query = args.place
    while True:
        if not query:
            if not interactive:
                raise SystemExit("Give a place name, e.g. finland2mc \"Jyväskylä\"")
            query = input("Place in Finland (town, district or address): ").strip()
            continue
        try:
            found = geocode.search(query)
        except OSError as e:
            raise SystemExit(f"Place search failed ({e}). Use --at LAT,LON instead.")
        if not found:
            say(f"Nothing found in Finland for '{query}'.")
            if not interactive:
                raise SystemExit(1)
            query = None
            continue
        if len(found) == 1 or not interactive:
            return found[0]
        for i, p in enumerate(found, 1):
            say(f"  {i}. {p.description}")
        while True:
            pick = ask("Which one", "1", interactive)
            if pick.isdigit() and 1 <= int(pick) <= len(found):
                return found[int(pick) - 1]
            say(f"Pick a number from 1 to {len(found)}.")


def folder_name(name):
    return re.sub(r'[<>"/\\|?*]', "", name.replace(":", "-")).strip() or "world"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="finland2mc", description="1:1 Minecraft worlds of places in Finland, "
                                 "from National Land Survey of Finland open data.")
    ap.add_argument("place", nargs="?", help="town, district or address in Finland")
    ap.add_argument("--at", metavar="LAT,LON", help="centre point instead of a place name "
                    "(or E381000,N6905000 in ETRS-TM35FIN)")
    ap.add_argument("--size", type=float, help="side of the square area in km (default 4, max 10)")
    ap.add_argument("--name", help="world name shown in Minecraft (default: '<place> 1:1')")
    ap.add_argument("--out", default="worlds", help="folder for the finished world (default: ./worlds)")
    ap.add_argument("--no-lidar", action="store_true",
                    help="skip laser scans: smaller download, but made-up tree and building heights")
    ap.add_argument("--prism", nargs="?", const=DEFAULT_INSTANCE, metavar="INSTANCE",
                    help=f"copy the world into a Prism Launcher instance (default '{DEFAULT_INSTANCE}', "
                         "created if missing)")
    ap.add_argument("--zip", action="store_true", help="also make a .zip of the world")
    ap.add_argument("--jobs", type=int, help="CPU cores to use (default: all but one)")
    ap.add_argument("--keep-work", action="store_true", help="keep the intermediate grids")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask anything, use defaults")
    ap.add_argument("--version", action="version", version=f"finland2mc {__version__} (Minecraft {MINECRAFT_VERSION})")
    args = ap.parse_args(argv)
    interactive = sys.stdin.isatty() and not args.yes

    place = choose_place(args, interactive)
    e, n = place.etrs
    say(f"Place: {place.description}\n  ETRS-TM35FIN E {e} N {n}")
    size = args.size
    while size is None:
        try:
            size = float(ask("Area size in km (one side of the square)", "4", interactive).replace(",", "."))
        except ValueError:
            say("Please give a number, e.g. 3 or 2.5")
    if not 0.5 <= size <= 10:
        raise SystemExit("Area size must be between 0.5 and 10 km.")
    name = args.name or f"{place.name} 1:1"
    area = Area(e, n, int(round(size * 500)))
    jobs = args.jobs or max(1, (os.cpu_count() or 2) - 1)

    say("Checking which map data is needed...")
    try:
        files, todo = nls.plan(area.bounds, lidar=not args.no_lidar)
    except ValueError as err:
        raise SystemExit(f"That point is outside Finland's map sheets ({err}).")
    if not files["dem"] and not files["mtk"]:
        raise SystemExit("No map data here. Is the place in Finland?")
    lidar_bytes = sum(nls.remote_size(p) or 0 for p in files["lidar"] if not nls.local(p).exists())
    if files["lidar"] and lidar_bytes > 0 and interactive:
        if not ask_yes(f"Use laser scans for real tree and building heights? ({lidar_bytes / 1e6:.0f} MB of the "
                       "download)", True, interactive):
            todo -= lidar_bytes
            files["lidar"] = []
    chunks = ((area.half - 1) // 16 + (area.half + 15) // 16 + 1) ** 2
    lo, hi = (chunks * s / jobs / 60 + size * size * 0.05 for s in SECONDS_PER_CHUNK)
    lo, hi = max(1, round(lo)), max(1, round(hi))
    say(f"World: '{name}', {size:g} x {size:g} km, {chunks} chunks, about {lo if lo == hi else f'{lo}-{hi}'} min "
        f"on {jobs} cores (forest is slower than town)")
    if not files["lidar"]:
        say("  (no laser scans: tree positions and building heights will be guesses)")
    say(f"Download: {todo / 1e6:.0f} MB (cached in {nls.CACHE}, reused next time)")

    install_to = args.prism
    root = prism.find_root()
    if install_to is None and root and interactive:
        if ask_yes(f"Install into Prism Launcher instance '{DEFAULT_INSTANCE}'?", True, interactive):
            install_to = DEFAULT_INSTANCE
    if not ask_yes("Start?", True, interactive):
        return 0

    t0 = time.time()
    if todo:
        say("Downloading National Land Survey data...")
    local = nls.download(files, progress=lambda m: print(f"\r{m}", end="", flush=True))
    print()

    out = Path(args.out).resolve()
    world = out / folder_name(name)
    if world.exists():
        raise SystemExit(f"{world} already exists. Remove it or use --name / --out.")
    work = nls.CACHE / "work" / f"{e}_{n}_{area.half}"
    from . import generate, prepare, render, world as world_mod  # heavy imports after the questions

    say("Preparing map data...")
    meta = prepare.Prep(area, local, work, log=lambda m: say("  " + m)).run()
    say("Building the world...")
    generate.run(str(work), str(world), jobs=jobs, progress=lambda m: print(f"\r{m}   ", end="", flush=True))
    print()
    spawn = world_mod.finalize(str(world), str(work), name)
    preview = Path(str(world) + ".png")
    render.render(str(world), str(preview), area.half)
    if args.zip:
        shutil.make_archive(str(world), "zip", out, world.name)
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    say(f"\nDone in {(time.time() - t0) / 60:.1f} min.")
    say(f"  World:   {world}")
    say(f"  Map:     {preview}")
    say(f"  Spawn:   {spawn[0]} {spawn[1]} {spawn[2]}  (y = {prepare.height_rule(meta['offset'])})")
    if install_to:
        try:
            inst, dest, created = prism.install(world, install_to)
            say(f"  Prism:   {'created instance' if created else 'added to instance'} '{install_to}' ({inst})")
            if created and prism.prism_running():
                say("           Prism is running; restart it if the new instance doesn't show up.")
        except (RuntimeError, OSError) as err:
            say(f"  Prism:   not installed ({err})")
    else:
        say(f"  Open it with Minecraft {MINECRAFT_VERSION} or newer: copy the world folder into your saves folder.")
    say("Contains data from the National Land Survey of Finland (CC BY 4.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
