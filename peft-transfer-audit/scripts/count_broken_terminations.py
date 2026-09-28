#!/usr/bin/env python3
"""Count broken-termination completions in GSM8K sample logs.

The LM Evaluation Harness writes two lines per problem in
samples_gsm8k_*.jsonl (one for strict-match, one for flexible-extract),
and the few-shot prompt inside each line already contains '####' markers.
This script therefore de-duplicates by doc_id and inspects only the raw
completion stored in resps[0][0].

A healthy run has exactly one '####' per completion (the final answer).
Multiple '####' markers mean the model kept generating after the final
answer (broken termination, failure mode 5 in the paper); they corrupt
the flexible-extract filter, which reads the last number in the output.

Usage:
    python scripts/count_broken_terminations.py results/pilot_b3_s43_rerun
"""
import glob
import json
import sys


def main():
    root = sys.argv[1]
    completions = {}
    for path in glob.glob(f"{root}/**/samples_gsm8k_*.jsonl", recursive=True):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec["doc_id"]
                if doc_id not in completions:
                    completions[doc_id] = rec["resps"][0][0]

    one = multi = none = overlong = 0
    for text in completions.values():
        n = text.count("####")
        if n == 1:
            one += 1
        elif n > 1:
            multi += 1
        else:
            none += 1
        if len(text) > 1500:
            overlong += 1

    total = len(completions)
    print(f"unique problems                    : {total}")
    print(f"exactly one '####' (clean stop)    : {one} ({100*one/total:.1f}%)")
    print(f"multiple '####' (broken stop)      : {multi} ({100*multi/total:.1f}%)")
    print(f"no '####'                          : {none} ({100*none/total:.1f}%)")
    print(f"overlong outputs (>1500 chars)     : {overlong} ({100*overlong/total:.1f}%)")


if __name__ == "__main__":
    main()
