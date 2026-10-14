"""Real-data sanity checks (no annotations needed).

self_before / self_after: one video against itself with disjoint keyframes
    (candidate sampling phase shifted by half a step) -> expect no changes.
swap: after vs before -> moved stays moved, added <-> removed.
"""
import json
import os
import sys
from pathlib import Path

from roomdelta.pipeline import Config, compare

V = Path("examples/room")
RUNS = {
    "self_before": (V / "before.mp4", V / "before.mp4", Config(phase_before=0, phase_after=5)),
    "self_after": (V / "after.mp4", V / "after.mp4", Config(phase_before=0, phase_after=5)),
    "swap": (V / "after.mp4", V / "before.mp4", Config()),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(RUNS)
    summary = {}
    for n in names:
        b, a, cfg = RUNS[n]
        out = Path(os.environ.get("ROOMDELTA_SANITY_OUT", "outputs/sanity")) / n
        compare(str(b), str(a), out, cfg)
        ev = json.load(open(out / "events.json"))
        summary[n] = {}
        for e in ev:
            summary[n].setdefault(e["event_type"], []).append(e["semantic_label"])
        print(n, {k: len(v) for k, v in summary[n].items()}, flush=True)
    json.dump(summary, open(Path(os.environ.get("ROOMDELTA_SANITY_OUT", "outputs/sanity")) / "summary.json", "w"), indent=1)
