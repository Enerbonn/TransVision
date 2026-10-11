"""Terminal batch worker: python -m src.cli --queue queue.json --output-dir output."""

import argparse
import sys
from pathlib import Path

from src.batch import BatchQueue


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="TransVision unattended sequential exports"
    )
    parser.add_argument(
        "--queue",
        required=True,
        help="Persistent queue JSON created in the editor or with --inputs",
    )
    parser.add_argument(
        "--inputs", help="JSON list of video paths or per-video crop settings"
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parents[1] / "output")
    )
    parser.add_argument("--task", choices=["pdf", "ocr", "both"], default="pdf")
    parser.add_argument("--sensitivity", type=int, default=50)
    parser.add_argument("--motion-mode", choices=["summary", "demo"], default="summary")
    parser.add_argument("--review-first", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    queue = BatchQueue(args.queue, args.output_dir)
    if args.inputs:
        import json

        queue.enqueue(
            json.loads(Path(args.inputs).read_text(encoding="utf-8")),
            {
                "task": args.task,
                "sensitivity": args.sensitivity,
                "motion_mode": args.motion_mode,
                "automatic_export": not args.review_first,
            },
        )
    if args.retry_failed:
        queue.retry()
    queue.run()
    state = queue.snapshot()
    for job in state["jobs"]:
        print(f"{job['name']}: {job['status']} — {job['message']}")
    return int(any(job["status"] == "failed" for job in state["jobs"]))


if __name__ == "__main__":
    raise SystemExit(main())
