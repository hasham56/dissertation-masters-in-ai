"""Read the Tiled `zones` object layer and write the viewer's layout as JSON.

    uv run python scripts/export_layout.py                 # viz/map/hamlet.tmj or .tmx -> viz/layout.json

Reads the object layer named ``zones``: rectangles named FARM, OFFICE, CANTEEN (see the alias
table), MARKET, SOCIAL and HOME_0..HOME_k.

**Positions only.** The layout says where things are on the map and nothing else. How many agents
there are, what each zone holds and what it is called all belong to the run, so they travel in the
replay and the viewer reads them from there. A layout that carried them would have to be rebuilt
every time the world changed, and a stale one serves N = 8 replays against an N = 9 layout that
prints the canteen's N = 9 name.

For each zone it precomputes the anchors the viewer draws on, so the viewer does no layout maths:

* ``active``: ``ANCHOR_SLOTS`` points on a grid inside the rectangle;
* ``queued``: ``ANCHOR_SLOTS`` points spread along the bottom edge, where refused agents wait;
* homes: one point each, since a home holds its own agent and nobody else.

Anchors are cut to fit the largest population the map serves rather than to a capacity, so one
layout serves every world the map can hold. A viewer showing N agents uses the first N; HOME_k
beyond a replay's population is simply never drawn.

Reads the map only; writes one JSON. Nothing under ``runs/`` is touched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_constants as K  # noqa: E402
from hamlet.config import ZONE_NAMES, HamletConfig  # noqa: E402

# A replay zone name may appear in the map under any of these, case-insensitively. Every other zone
# matches exactly. CANTEEN is the code identifier; the map has carried FOOD_STREET and the display
# name is Restaurant, so all three resolve to the same rectangle.
ALIASES: dict[str, set[str]] = {
    "CANTEEN": {"CANTEEN", "RESTAURANT", "FOOD_STREET"},
}
# Anchors drawn per zone. Cut to the largest population the map is meant to serve rather than to a
# capacity, so one layout serves every world the map can hold: a replay with fewer agents uses the
# first few and leaves the rest unused. Raise this only alongside a map that has the homes for it.
ANCHOR_SLOTS = 8


def parse_tmx(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = ET.parse(path).getroot()
    meta = {"width": int(root.get("width")), "height": int(root.get("height")),
            "tilewidth": int(root.get("tilewidth")), "tileheight": int(root.get("tileheight"))}
    layer = next((g for g in root.iter("objectgroup") if g.get("name") == "zones"), None)
    if layer is None:
        raise SystemExit(f"{path} has no object layer named 'zones'")
    objects = []
    for o in layer.iter("object"):
        props = {p.get("name"): p.get("value") for p in o.iter("property")}
        objects.append({"name": o.get("name"), "x": float(o.get("x")), "y": float(o.get("y")),
                        "width": float(o.get("width", 0)), "height": float(o.get("height", 0)),
                        "properties": props})
    return meta, objects


def parse_tmj(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    j = json.loads(path.read_text())
    meta = {k: j[k] for k in ("width", "height", "tilewidth", "tileheight")}
    layer = next((l for l in j.get("layers", []) if l.get("name") == "zones"), None)
    if layer is None:
        raise SystemExit(f"{path} has no object layer named 'zones'")
    objects = []
    for o in layer.get("objects", []):
        props = {p["name"]: p["value"] for p in o.get("properties", [])}
        objects.append({"name": o.get("name"), "x": float(o["x"]), "y": float(o["y"]),
                        "width": float(o.get("width", 0)), "height": float(o.get("height", 0)),
                        "properties": props})
    return meta, objects


def grid_points(x: float, y: float, w: float, h: float, n: int) -> list[list[float]]:
    """``n`` points on a small grid in the LOWER THIRD of the rectangle.

    The buildings are drawn with their roofs occupying the upper part of each rectangle, so an
    anchor placed in the middle would put a sprite on a roof. Keeping the grid in the bottom third
    puts agents on the ground in front of the building, which is also where a viewer expects them.
    """
    if n <= 0:
        return []
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))
    band_top = y + h * (2.0 / 3.0)              # the lower third
    band_h = max(h / 3.0, 1.0)
    pad_x = w * 0.12
    ix, iw = x + pad_x, max(w - 2 * pad_x, 1.0)
    pts = []
    for k in range(n):
        r, c = divmod(k, cols)
        cx = ix + (iw * (c + 0.5) / cols)
        cy = band_top + (band_h * (r + 0.5) / rows)
        pts.append([round(cx, 1), round(cy, 1)])
    return pts


def edge_points(x: float, y: float, w: float, h: float, n: int) -> list[list[float]]:
    """``n`` points spread along the bottom edge, where refused agents queue."""
    if n <= 0:
        return []
    pad = w * 0.1
    ix, iw = x + pad, max(w - 2 * pad, 1.0)
    return [[round(ix + iw * (k + 0.5) / n, 1), round(y + h, 1)] for k in range(n)]


def build(map_path: Path, anchor_slots: int = ANCHOR_SLOTS) -> dict[str, Any]:
    meta, objects = (parse_tmj(map_path) if map_path.suffix == ".tmj" else parse_tmx(map_path))
    by_upper = {str(o["name"]).upper(): o for o in objects if o.get("name")}

    zones: dict[str, Any] = {}
    homes: list[dict[str, Any]] = []
    missing: list[str] = []
    drift: list[str] = []

    for zone in ZONE_NAMES:
        if zone in ("TRANSIT", "HOME"):
            continue
        names = ALIASES.get(zone, {zone})
        found = next((by_upper[n] for n in sorted(names) if n in by_upper), None)
        if found is None:
            missing.append(zone)
            continue
        x, y, w, h = found["x"], found["y"], found["width"], found["height"]
        zones[zone] = {
            "map_name": found["name"],
            "rect": [x, y, w, h],
            "centre": [round(x + w / 2, 1), round(y + h / 2, 1)],
            "active": grid_points(x, y, w, h, anchor_slots),
            "queued": edge_points(x, y, w, h, anchor_slots),
        }

    # Every HOME_k the map defines becomes a home. A replay with fewer agents leaves the surplus
    # homes undrawn; one with more is refused by the viewer, not silently squeezed in here.
    home_ids = sorted(int(k.removeprefix("HOME_")) for k in by_upper if k.startswith("HOME_")
                      and k.removeprefix("HOME_").isdigit())
    for i in home_ids:
        o = by_upper[f"HOME_{i}"]
        x, y, w, h = o["x"], o["y"], o["width"], o["height"]
        homes.append({
            "agent": i,
            "map_name": o["name"],
            "rect": [x, y, w, h],
            "centre": [round(x + w / 2, 1), round(y + h / 2, 1)],
            # a home holds its own agent and nobody else: one anchor, at the bottom centre
            "active": [[round(x + w / 2, 1), round(y + h, 1)]],
            "queued": [[round(x + w / 2, 1), round(y + h, 1)]],
        })

    if missing:
        raise SystemExit(
            "the map has no rectangle for: " + ", ".join(missing)
            + "\nnames present: " + ", ".join(sorted(by_upper))
            + "\nadd the rectangle in Tiled, or extend ALIASES in scripts/export_layout.py")

    if drift:
        print("  note: the map's capacity properties disagree with the simulation and were ignored:")
        for d in drift:
            print(f"    {d}")
    return {"source": str(map_path), "map": meta, "anchor_slots": ANCHOR_SLOTS,
            "homes_defined": [h["agent"] for h in homes],
            "aliases": {k: sorted(v) for k, v in ALIASES.items()},
            "zones": zones, "homes": homes}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default=None, help="the Tiled map (default: hamlet.tmj, else hamlet.tmx)")
    ap.add_argument("--out", default="viz/layout.json")
    args = ap.parse_args(argv)

    if args.map:
        map_path = Path(args.map)
    else:
        # Tiled writes .tmx and exports .tmj, and the two drift apart whenever only one is saved.
        # Take whichever is newer rather than always preferring .tmj: a .tmj left behind by a later
        # .tmx save would otherwise be read without a word.
        found = [q for q in (Path("viz/map/hamlet.tmj"), Path("viz/map/hamlet.tmx")) if q.exists()]
        map_path = max(found, key=lambda q: q.stat().st_mtime) if found else Path("viz/map/hamlet.tmj")
    if not map_path.exists():
        raise SystemExit(f"no map at {map_path}")

    layout = build(map_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(layout, indent=1))
    print(f"  read {map_path} ({map_path.suffix[1:].upper()})")
    for zone, z in layout["zones"].items():
        note = f' via alias "{z["map_name"]}"' if z["map_name"].upper() != zone else ""
        print(f'  {zone:8s}{note}: {len(z["active"])} active and {len(z["queued"])} queued anchor(s)')
    print(f"  homes defined by the map: {layout['homes_defined']}")
    print("  capacities and printed names are not written here; they travel in the replay")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
