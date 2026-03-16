#!/usr/bin/env python3
"""
F1 Constants Cracker — Cliff Physics Model
Run: python crack_constants.py
Output: solution/constants.json
"""

import json, os, glob, sys, time
import numpy as np
from scipy.optimize import differential_evolution

def calc_lap_time(base_time, compound, age, track_temp, params):
    o_soft, o_hard, d_soft, d_med, d_hard, w_soft, w_med, w_hard, c_soft, c_med, c_hard, t_coeff, t_base = params
    compound_base = {"SOFT": o_soft, "MEDIUM": 0.0, "HARD": o_hard}
    degrade_slope = {"SOFT": d_soft, "MEDIUM": d_med, "HARD": d_hard}
    cliff_start   = {"SOFT": round(w_soft), "MEDIUM": round(w_med), "HARD": round(w_hard)}
    cliff_slope   = {"SOFT": c_soft, "MEDIUM": c_med, "HARD": c_hard}
    temp_factor   = 1.0 + (track_temp - t_base) * t_coeff
    age_term      = age * degrade_slope[compound] * temp_factor
    cliff_laps    = max(0, age - cliff_start[compound])
    cliff_term    = cliff_laps * cliff_slope[compound] * temp_factor
    return base_time + compound_base[compound] + age_term + cliff_term

def get_race_error(race_data, params):
    config       = race_data["race_config"]
    real_order   = race_data["finishing_positions"]
    driver_times = {}
    for _, strategy in race_data["strategies"].items():
        total_time   = 0.0
        current_tire = strategy["starting_tire"]
        tire_age     = 0
        pit_stops    = {int(s["lap"]): s["to_tire"] for s in strategy.get("pit_stops", [])}
        for lap in range(1, int(config["total_laps"]) + 1):
            tire_age   += 1
            total_time += calc_lap_time(float(config["base_lap_time"]), current_tire, tire_age, float(config["track_temp"]), params)
            if lap in pit_stops:
                total_time  += float(config["pit_lane_time"])
                current_tire = pit_stops[lap]
                tire_age     = 0
        driver_times[strategy["driver_id"]] = total_time
    error = 0.0
    for i in range(len(real_order) - 1):
        t_a = driver_times[real_order[i]]
        t_b = driver_times[real_order[i+1]]
        if t_a >= t_b:
            error += (t_a - t_b) + 1.0
    return error

def objective(params, batch):
    return sum(get_race_error(r, params) for r in batch) / len(batch)

def check_accuracy(params, races, limit=500):
    correct = 0
    for race in races[:limit]:
        config = race["race_config"]
        times  = {}
        for _, strat in race["strategies"].items():
            t = 0.0; tire = strat["starting_tire"]; age = 0
            pits = {int(s["lap"]): s["to_tire"] for s in strat.get("pit_stops", [])}
            for lap in range(1, int(config["total_laps"]) + 1):
                age += 1
                t   += calc_lap_time(float(config["base_lap_time"]), tire, age, float(config["track_temp"]), params)
                if lap in pits:
                    t += float(config["pit_lane_time"]); tire = pits[lap]; age = 0
            times[strat["driver_id"]] = t
        if sorted(times.keys(), key=lambda d: times[d]) == race["finishing_positions"]:
            correct += 1
    return correct / min(limit, len(races)) * 100

PARAM_NAMES = ["o_soft","o_hard","d_soft","d_med","d_hard","w_soft","w_med","w_hard","c_soft","c_med","c_hard","t_coeff","t_base"]

BOUNDS = [
    (-0.75,-0.50),(0.25, 0.40),
    (0.08, 0.12), (0.02, 0.05),(0.01,0.03),
    (8,    12),   (18,   22),  (30,  36),
    (0.08, 0.12), (0.05, 0.08),(0.04,0.07),
    (0.02, 0.04), (18.0, 22.0)
]

if __name__ == "__main__":
    print(f"{'='*60}\n  F1 Cliff Physics Cracker | CPUs: {os.cpu_count()}\n{'='*60}")
    files = sorted(glob.glob("data/historical_races/races_*.json"))
    if not files:
        print("ERROR: No race files found."); sys.exit(1)
    print(f"  Found {len(files)} race files\n")

    best_params = None
    best_error  = float("inf")
    bounds      = BOUNDS
    t_start     = time.time()

    for i, fpath in enumerate(files):
        print(f"\n{'='*60}\n  FILE {i+1}/{len(files)}: {os.path.basename(fpath)} | {time.time()-t_start:.0f}s elapsed\n{'='*60}")
        batch  = json.load(open(fpath))
        result = differential_evolution(
            objective, bounds, args=(batch,),
            maxiter=100, popsize=25, mutation=(0.5,1.0), recombination=0.9,
            workers=-1, updating="deferred", seed=42, disp=True, tol=1e-10,
        )
        p   = result.x
        err = result.fun
        acc = check_accuracy(p, batch)
        print(f"\n  Avg Error: {err:.8f} | Accuracy: {acc:.1f}%")

        if err < best_error:
            best_error = err; best_params = p.copy()

        bounds = [(v*0.98, v*1.02) for v in p]

        os.makedirs("solution", exist_ok=True)
        out = dict(zip(PARAM_NAMES, best_params.tolist()))
        out["_best_error"] = float(best_error); out["_files_done"] = i+1
        json.dump(out, open("solution/constants.json","w"), indent=2)
        print(f"  Saved → solution/constants.json")

        if err <= 0.000001:
            print("\n🎯 PERFECT! Stopping early."); break

    print(f"\n{'='*60}\n  FINAL PARAMS\n{'='*60}")
    for n, v in zip(PARAM_NAMES, best_params): print(f"  {n:<12} = {v:.8f}")
    all_r = []
    for fp in files[:2]: all_r.extend(json.load(open(fp)))
    final_acc = check_accuracy(best_params, all_r, 1000)
    print(f"\nFinal accuracy (1000 races): {final_acc:.1f}%")
    out = dict(zip(PARAM_NAMES, best_params.tolist()))
    out["_final_accuracy"] = final_acc
    json.dump(out, open("solution/constants.json","w"), indent=2)
    print(f"Total time: {time.time()-t_start:.0f}s")
