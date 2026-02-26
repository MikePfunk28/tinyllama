"""
Simple Continuous Loop - Works correctly
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

LOG = "M:/tinyllama/loop.log"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=600):
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
    path = "M:/tinyllama/experts/quality_results.json"
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
            return d.get("results", {})
    return {}


def main():
    log("=" * 50)
    log("SIMPLE LOOP - Target 95%")
    log("=" * 50)

    it = 0
    while True:
        it += 1
        log(f"\n### ITERATION {it} ###")

        # 1. Quality check
        log("1. Quality check")
        run("quick_quality.py", 120)

        q = get_quality()
        if q:
            avg = sum(q.values()) / len(q) if q else 0
            log(f"   Avg: {avg * 100:.0f}%")
            for d, v in sorted(q.items(), key=lambda x: x[1]):
                log(f"   {d}: {v * 100:.0f}%")

        # 2. Smart train (shorter)
        log("2. Smart train")
        run("smart_train.py", 300)

        # 3. Router
        log("3. Router")
        run("train_router_only.py", 120)

        # 4. Test
        log("4. Test")
        run("agent.py", 60)

        # Check goal
        q = get_quality()
        if q and all(v >= 0.95 for v in q.values()):
            log("\n!!! GOAL REACHED - 95% !!!")
            break

        log(f"\nWaiting 2 min...")
        time.sleep(120)


if __name__ == "__main__":
    main()
