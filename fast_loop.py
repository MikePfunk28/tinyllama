"""
Simple Fast Loop - Quick iterations
"""

import os
import json
import time
import subprocess
from datetime import datetime

LOG = "M:/tinyllama/loop.log"
QUAL = "M:/tinyllama/experts/quality_results.json"
GOAL = 0.95


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=300):
    log(f"  > {cmd}")
    try:
        r = subprocess.run(
            ["python", f"M:/tinyllama/{cmd}"],
            timeout=timeout,
            cwd="M:/tinyllama",
            capture_output=True,
            text=True,
        )
        ok = r.returncode == 0
        log(f"    {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        log(f"    ERR: {e}")
        return False


def get_quality():
    if os.path.exists(QUAL):
        with open(QUAL) as f:
            d = json.load(f)
            return d.get("results", {})
    return {}


def main():
    log("=" * 50)
    log("FAST LOOP - Target 95%")
    log("=" * 50)

    it = 0
    while True:
        it += 1
        log(f"\n### ITERATION {it} ###")

        # 1. Quality
        log("1. Quality")
        run("better_quality.py", 300)

        q = get_quality()
        if q:
            avg = sum(q.values()) / len(q)
            log(f"   Avg: {avg * 100:.0f}%")
            for d, v in sorted(q.items(), key=lambda x: x[1]):
                st = "OK" if v >= GOAL else "LOW"
                log(f"   {d}: {v * 100:.0f}% [{st}]")

        # Check goal
        if q and all(v >= GOAL for v in q.values()):
            log("\n!!! GOAL REACHED !!!")
            break

        # 2. Fast train
        log("2. Fast train")
        run("fast_train.py", 600)

        # 3. Router
        log("3. Router")
        run("train_router_only.py", 120)

        # 4. Test
        log("4. Test")
        run("agent.py", 60)

        log(f"\nWait 1 min...")
        time.sleep(60)


if __name__ == "__main__":
    main()
