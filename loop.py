"""
Continuous Improvement Loop
Runs forever: fetch data -> train -> evaluate -> repeat
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

LOG_FILE = "M:/tinyllama/continuous.log"


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_script(script, timeout=600):
    """Run a script and return success"""
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
        else:
            log(f"  Failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        log(f"  Timeout")
        return False
    except Exception as e:
        log(f"  Error: {e}")
        return False


def get_stats():
    """Get current stats"""
    stats = {"adapters": 0, "samples": 0, "router_acc": 0}

    # Count adapters
    for f in os.listdir("M:/tinyllama/experts"):
        if f.endswith("_lora_adapter.pt"):
            stats["adapters"] += 1

    # Count samples
    data_dir = "M:/tinyllama/cleaned_data"
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith("_train.json"):
                path = os.path.join(data_dir, f)
                try:
                    with open(path) as fp:
                        data = json.load(fp)
                        stats["samples"] += len(data)
                except:
                    pass

    # Get router accuracy
    router_path = "M:/tinyllama/experts/lightweight_router.pt"
    if os.path.exists(router_path):
        import torch

        try:
            ckpt = torch.load(router_path, map_location="cpu", weights_only=False)
            stats["router_acc"] = ckpt.get("accuracy", 0)
        except:
            pass

    return stats


def main():
    log("=" * 60)
    log("STARTING CONTINUOUS IMPROVEMENT LOOP")
    log("=" * 60)

    iteration = 0

    while True:
        iteration += 1
        log(f"\n{'#' * 60}")
        log(f"ITERATION {iteration}")
        log(f"{'#' * 60}")

        # Phase 1: Fetch new data
        log("\n--- Phase 1: Fetch Data ---")
        run_script("M:/tinyllama/fetch_data.py", timeout=300)

        # Phase 2: Train LoRAs
        log("\n--- Phase 2: Train LoRAs ---")
        run_script("M:/tinyllama/train_loras.py", timeout=1800)

        # Phase 3: Train Router
        log("\n--- Phase 3: Train Router ---")
        run_script("M:/tinyllama/train_router_only.py", timeout=120)

        # Phase 4: Test
        log("\n--- Phase 4: Test ---")
        run_script("M:/tinyllama/agent.py", timeout=120)

        # Stats
        stats = get_stats()
        log(f"\n--- Stats ---")
        log(f"  Adapters: {stats['adapters']}")
        log(f"  Total samples: {stats['samples']}")
        log(f"  Router accuracy: {stats['router_acc']:.2%}")

        log(f"\n--- Iteration {iteration} Complete ---")
        log(f"Waiting 5 minutes before next iteration...")
        log(f"Press Ctrl+C to stop")

        try:
            time.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            log("Stopped by user")
            break

    log("\nCONTINUOUS LOOP ENDED")


if __name__ == "__main__":
    main()
