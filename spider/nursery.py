"""Headless measurement: run one policy for a number of moments and print a line per block.

    python -m spider.nursery --moments 3000 --policy brain --seed 0
    python -m spider.nursery --moments 3000 --policy random --seed 0
    python -m spider.nursery --moments 3000 --policy frozen --brain runs/spider.npz
    python -m spider.nursery --moments 3000 --policy brain --gene modules=48,24 --gene arousal.need=0

The world's reward is the only teacher. Random is the baseline; frozen (a newborn, or a saved
brain that answers greedily and never learns) is the control for "kept learning". ``--brain``
with ``--policy brain`` continues a saved life, with its world and pending outcome when the
sidecar is beside it. Everything runs on the CPU.

The body can fall. A block reports the share of its moments upright (the torso's up axis above
``UPRIGHT`` along the world's) and the falls so far; a fallen life is a reading too, and the
world is never reset to help it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from .brain import DEFAULT_LAYOUT, LAYOUTS
from .brain import genes_from, parse_gene  # noqa: F401  (the server imports them from here too)
from .life import (BACKWARD_PAY, BLOCK, COURSE, HIGH_WATER, OLD_GROUND, PROGRESS_PAY, PROGRESS_WINDOW, STANCE_PAY,
                   STRICT, UPRIGHT, YAW_PAY, Life, checked_course)
from .model import MODEL_DIGEST
from .senses import PER, Senses


HEAD = (f"{'moment':>7} {'forward_m':>9} {'r/moment':>8} {'aroused':>7} {'sweeps':>6} "
        f"{'learn':>6} {'refused':>7} {'ms':>6} {'stance':>6} {'turned':>7} {'course_m':>8} {'from_birth':>10} "
        f"{'upright':>7} {'falls':>5}")


def law_arguments(parser: argparse.ArgumentParser, default=lambda name, value: value) -> None:
    """The law's flags, shared by the nursery and the server; ``default`` lets the server's
    environment set each one's default."""
    parser.add_argument("--stance-pay", type=float, default=float(default("STANCE_PAY", STANCE_PAY)),
                        help=f"reward taken per foot short of half the feet planted (default {STANCE_PAY}; 0 = off)")
    parser.add_argument("--yaw-pay", type=float, default=float(default("YAW_PAY", YAW_PAY)),
                        help=f"reward taken per radian turned in a moment (default {YAW_PAY}; 0 = off)")
    parser.add_argument("--course", type=checked_course, default=checked_course(default("COURSE", COURSE)),
                        help="pay progress along this fixed heading of the world (radians; 'birth' or 0 is the heading at "
                             "birth) instead of along the body's own heading (default off)")
    parser.add_argument("--strict", action="store_true", default=default("STRICT", STRICT),
                        help="the strict course: under --course, pay the lesser of the motion along the course and along "
                             "the body's own heading, so that only nose first on the course pays in full (default off)")
    parser.add_argument("--window", type=int, default=int(default("PROGRESS_WINDOW", PROGRESS_WINDOW)),
                        help=f"moments the progress term is averaged over (default {PROGRESS_WINDOW})")
    parser.add_argument("--high-water", action="store_true", default=default("HIGH_WATER", HIGH_WATER),
                        help="pay new ground only: progress beyond the farthest point reached (default off)")
    parser.add_argument("--old-ground", type=float, default=float(default("OLD_GROUND", OLD_GROUND)),
                        help=f"under --high-water, the share paid for progress on old ground, 0 to 1 (default {OLD_GROUND})")
    parser.add_argument("--pay-per-metre", type=float, default=float(default("PAY_PER_METRE", PROGRESS_PAY)),
                        help=f"reward per metre of progress, under the cap of 0.5 a moment (default {PROGRESS_PAY})")
    parser.add_argument("--backward-pay", type=float, default=float(default("BACKWARD_PAY", BACKWARD_PAY)),
                        help=f"share of the pay per metre charged for backward progress, 0 to 1 (default {BACKWARD_PAY}; 0 = free)")


def birth_arguments(parser: argparse.ArgumentParser, default=lambda name, value: value) -> None:
    """A newborn's flags (a saved life keeps its own), shared by the nursery and the server."""
    parser.add_argument("--gene", type=parse_gene, action="append", default=[],
                        help="name=value for a newborn brain, e.g. modules=48,24 or arousal.need=0 (repeatable)")
    parser.add_argument("--per", type=int, default=None, help=f"tuned inputs per reading (default {PER})")
    parser.add_argument("--compass", action="store_true", default=default("COMPASS", False),
                        help="sense the heading too, its cosine and sine (two readings more): the sense the course law is read by")
    parser.add_argument("--ground", action="store_true", default=default("GROUND", False),
                        help="sense the velocity over the ground in the world's frame too (two readings more): what the course law pays, read from the world")
    parser.add_argument("--layout", choices=LAYOUTS, default=default("LAYOUT", DEFAULT_LAYOUT),
                        help=f"how a newborn brain's answer is read: a slot per joint (three answers), per leg "
                             f"(twenty-seven, the triples of its three joints' answers) or per diagonal pair of legs (twenty-seven, both "
                             f"legs the same answer; a coupled body, a different model); a saved brain keeps its own "
                             f"(default {DEFAULT_LAYOUT})")
    parser.add_argument("--backend", default=default("BACKEND", None) or None,
                        help="cpu (or auto, the same): this project runs on the CPU only")


def life_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """The Life arguments the shared flags name."""
    return dict(
        genes=genes_from(args.gene) or None, backend=args.backend,
        senses=Senses(per=args.per or PER, compass=args.compass, ground=args.ground),
        stance_pay=args.stance_pay, yaw_pay=args.yaw_pay, window=args.window, high_water=args.high_water,
        old_ground=args.old_ground, pay_per_metre=args.pay_per_metre, backward_pay=args.backward_pay,
        course=args.course, strict=args.strict, layout=args.layout,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = life_kwargs(args)
    runs = args.save.parent if args.save else Path("runs")
    name = args.save.stem if args.save else "spider"
    continuing = bool(args.brain and args.policy == "brain" and args.brain.with_suffix(".json").exists())
    life = Life(args.policy, seed=args.seed, pace=0.0, runs=args.brain.parent if continuing else runs,
                brain_path=None if continuing else args.brain,
                name=args.brain.stem if continuing else name, **kwargs)
    life.autosave_minutes = 0.0
    if continuing:
        life.load()
        life.runs, life.name = runs, name
    started = life.moments
    falls_before = life.falls
    block = max(1, int(args.block))
    began = time.perf_counter()
    rows: list[dict[str, Any]] = []
    window: list[dict[str, Any]] = []
    mark = life.distance_forward
    course_mark = life.distance_course
    turned_mark = life.turned
    place = list(life.displacement)          # where the run starts, for the run's own displacement
    upright = 0
    if not args.quiet:
        print(HEAD, flush=True)
    for _ in range(args.moments):
        window.append(life.tick())
        if len(window) == block:
            up = sum(1 for w in window if w["upright"] > UPRIGHT)
            upright += up
            row = {
                "moment": life.moments,
                "forward_m": round(life.distance_forward - mark, 4),
                "reward_per_moment": round(sum(w["reward"] or 0.0 for w in window) / block, 4),
                "aroused_share": round(sum(1 for w in window if w["mode"] == "aroused") / block, 3),
                "sweeps_per_moment": round(sum(w["sweeps"] for w in window) / block, 2),
                "learning_sweeps": int(sum(w["learning_sweeps"] for w in window)),
                "refused": int(sum(1 for w in window if w["refused"])),
                "ms_per_moment": round(sum(w["ms"] for w in window) / block, 2),
                "mean_stance": round(sum(sum(w["pose"]["planted"]) for w in window) / block, 2),
                "turned_rad": round(life.turned - turned_mark, 3),
                "course_m": round(life.distance_course - course_mark, 4),
                "displacement_m": round(life.displacement_m(), 4),      # from the birth point, at the block's end
                "upright_share": round(up / block, 3),
                "falls": life.falls - falls_before,                     # over the run, at the block's end
            }
            rows.append(row)
            mark = life.distance_forward
            course_mark = life.distance_course
            turned_mark = life.turned
            window = []
            if not args.quiet:
                print(f"{row['moment']:>7} {row['forward_m']:>9.3f} {row['reward_per_moment']:>8.4f} "
                      f"{row['aroused_share']:>7.2f} {row['sweeps_per_moment']:>6.1f} {row['learning_sweeps']:>6} "
                      f"{row['refused']:>7} {row['ms_per_moment']:>6.2f} {row['mean_stance']:>6.2f} "
                      f"{row['turned_rad']:>7.2f} {row['course_m']:>8.3f} {row['displacement_m']:>10.3f} "
                      f"{row['upright_share']:>7.2f} {row['falls']:>5}", flush=True)
    elapsed = time.perf_counter() - began
    half = len(rows) // 2
    counted = block * len(rows)
    summary: dict[str, Any] = {
        "policy": args.policy,
        "seed": life.seed,
        "moments": args.moments,
        "from_moment": started,
        "forward_m": round(sum(r["forward_m"] for r in rows), 4),
        "course_m": round(sum(r["course_m"] for r in rows), 4),
        # straight-line metres: from the point of birth at the end, and over this run alone (the
        # same on a newborn); a circle walked for forward metres shows here as nothing
        "displacement_m": round(life.displacement_m(), 4),
        "run_displacement_m": round(math.hypot(life.displacement[0] - place[0], life.displacement[1] - place[1]), 4),
        "first_half_m": round(sum(r["forward_m"] for r in rows[:half]), 4),
        "second_half_m": round(sum(r["forward_m"] for r in rows[half:]), 4),
        "upright_share": round(upright / counted, 3) if counted else None,
        "falls": life.falls - falls_before,
        "upright_at_end": life.body.upright > UPRIGHT,
        "refused": int(sum(r["refused"] for r in rows)),
        "turned_rad": round(sum(r["turned_rad"] for r in rows), 3),
        "ms_per_moment": round(1000.0 * elapsed / max(1, args.moments), 3),
        "seconds": round(elapsed, 1),
        "placement": life.brain.describe()["placement"] if life.brain is not None else None,
        "genes": life.genes or {},
        **life.law(),                       # a continued life's own law, whatever the flags said
        "law_changes": life.law_changes,
        "model": MODEL_DIGEST,              # the body every reading is of
        "layout": life.layout,
        "per": life.senses.per,
        "compass": life.senses.compass,
        "ground": life.senses.ground,
        "blocks": rows,
    }
    if args.save and life.brain is not None:
        summary["saved"] = life.save()
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--moments", type=int, default=3000)
    parser.add_argument("--policy", choices=("brain", "random", "frozen"), default="brain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=BLOCK)
    parser.add_argument("--brain", type=Path, default=None, help="a saved brain to continue, or to freeze")
    parser.add_argument("--save", type=Path, default=None, help="save the brain and its world here after the run (.npz)")
    birth_arguments(parser)
    law_arguments(parser)
    parser.add_argument("--json", type=Path, default=None, help="write the summary here")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.save and args.json and args.json.resolve() == args.save.with_suffix(".json").resolve():
        parser.error("--json would overwrite the saved brain's sidecar; choose another name")
    summary = run(args)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=1))
    if not args.quiet:
        print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=1))
    return summary


if __name__ == "__main__":
    main()
    sys.exit(0)
