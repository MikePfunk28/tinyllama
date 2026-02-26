"""
Enhanced Continuous Loop
- Evaluates quality before each iteration
- Only trains domains below 95%
- More aggressive training for weak domains
- Tracks progress toward 95% goal
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

LOG_FILE = "M:/tinyllama/continuous.log"
QUALITY_FILE = "M:/tinyllama/experts/quality_results.json"
GOAL = 0.95


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_script(script, timeout=1800):
    log(f"Running: {script}")
    try:
        result = subprocess.run(
            ["python", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="M:/tinyllama",
        )
        if result.returncode == 0:
            log(f"  Success")
            return True
        err = result.stderr[:200] if result.stderr else result.stdout[:200]
        log(f"  Failed: {err}")
        return False
    except Exception as e:
        log(f"  Error: {e}")
        return False
    except Exception as e:
        log(f"  Error: {e}")
        return False
    except Exception as e:
        log(f"  Error: {e}")
        return False


def get_quality():
    if os.path.exists(QUALITY_FILE):
        with open(QUALITY_FILE) as f:
            data = json.load(f)
            return data.get("results", {})
    return {}


def check_goal_reached():
    quality = get_quality()
    if not quality:
        return False
    return all(
        quality.get(d, 0) >= GOAL
        for d in ["math", "code", "reasoning", "conversation", "knowledge", "science"]
    )


def main():
    log("=" * 60)
    log("ENHANCED CONTINUOUS LOOP - Target 95%")
    log("=" * 60)

    iteration = 0

    while True:
        iteration += 1
        log(f"\n{'#' * 60}")
        log(f"ITERATION {iteration}")
        log(f"{'#' * 60}")

        # Phase 1: Evaluate current quality
        log("\n--- Phase 1: Evaluate Quality ---")
        run_script("M:/tinyllama/evaluate_quality.py", timeout=300)

        quality = get_quality()
        if quality:
            log("\n  Quality scores:")
            for d, q in sorted(quality.items(), key=lambda x: x[1]):
                bar = "█" * int(q * 20) + "░" * (20 - int(q * 20))
                status = "OK" if q >= GOAL else "TRAIN"
                log(f"    {d}: [{bar}] {q * 100:.1f}% [{status}]")

            avg = sum(quality.values()) / len(quality) if quality else 0
            log(f"\n  Average: {avg * 100:.1f}% | Goal: {GOAL * 100:.0f}%")

        # Check if goal reached
        if check_goal_reached():
            log("\n" + "=" * 60)
            log("GOAL REACHED! All domains at 95%+")
            log("=" * 60)
            break

        # Phase 2: Fetch more data
        log("\n--- Phase 2: Fetch Data ---")
        run_script("M:/tinyllama/fetch_data.py", timeout=300)

        # Phase 3: Smart training (only domains below 95%)
        log("\n--- Phase 3: Smart Training ---")
        run_script("M:/tinyllama/smart_train.py", timeout=1800)

        # Phase 4: Train router
        log("\n--- Phase 4: Train Router ---")
        run_script("M:/tinyllama/train_router_only.py", timeout=120)

        # Phase 5: Final evaluation
        log("\n--- Phase 5: Final Evaluation ---")
        run_script("M:/tinyllama/evaluate_quality.py", timeout=300)

        # Stats
        final_quality = get_quality()
        if final_quality:
            avg = sum(final_quality.values()) / len(final_quality)
            domains_above = sum(1 for q in final_quality.values() if q >= GOAL)
            log(f"\n--- Iteration {iteration} Summary ---")
            log(f"  Average quality: {avg * 100:.1f}%")
            log(f"  Domains at 95%+: {domains_above}/6")
            log(f"  Progress: {avg * 100 / 95:.1f}% of goal")

        log(f"\n--- Waiting 3 minutes ---")
        log("Press Ctrl+C to stop")

        try:
            time.sleep(180)  # 3 minutes
        except KeyboardInterrupt:
            log("Stopped by user")
            break

    log("\nLOOP ENDED")


if __name__ == "__main__":
    main()
