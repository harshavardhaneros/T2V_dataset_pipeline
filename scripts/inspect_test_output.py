#!/usr/bin/env python3
"""Quick sanity check for test pipeline workspace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True, help="e.g. workspaces/ABCD_test_100_130")
    p.add_argument("--show-captions", action="store_true")
    args = p.parse_args()

    ws = Path(args.workspace)
    meta = ws / "metadata.jsonl"
    if not meta.exists():
        print(f"MISSING {meta}")
        return

    records = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
    print(f"workspace: {ws}")
    print(f"clips: {len(records)}")
    print(f"keep: {sum(1 for r in records if r.get('keep'))}")
    print(f"verdicts: {dict(Counter(r.get('verdict') for r in records))}")
    print(f"actor_status: {dict(Counter(r.get('actor_status') for r in records))}")
    print(f"buckets (top 5): {Counter(r.get('bucket') for r in records).most_common(5)}")

    tagged = [r for r in records if r.get("actors")]
    print(f"clips with actors: {len(tagged)}")
    if tagged:
        t = tagged[0]
        print(f"  sample: {t['clip_id']} -> {t['actors'][0].get('display_name')}")

    captioned = [r for r in records if r.get("caption")]
    print(f"clips with caption: {len(captioned)}")
    if args.show_captions and captioned:
        for r in captioned[:3]:
            print(f"\n--- {r['clip_id']} [{r.get('timestamp_start')}-{r.get('timestamp_end')}s] ---")
            print(f"actors: {[a.get('display_name') for a in r.get('actors', [])]}")
            print(r.get("caption", "")[:400])


if __name__ == "__main__":
    main()
