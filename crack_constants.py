#!/usr/bin/env python3
"""
F1 Constants Cracker - Box Box Box Challenge
============================================
Reverse-engineers the 7 hidden simulation constants from historical race data.

KEY IMPROVEMENTS OVER ORIGINAL:
  1. Full pairwise ranking loss (not just adjacent pairs) - catches all inversions
  2. scipy.differential_evolution(workers=-1) → ALL CPU cores in parallel automatically
  3. Numpy-vectorized stint precomputation - 10-50x faster objective evaluation
  4. Two-phase: DE global search → L-BFGS-B precision polish
  5. Latin hypercube init for better parameter space coverage
  6. Diverse race sampling across tracks and temperatures
  7. S_delta fixed at 0 (SOFT is reference) - removes 1 redundant param

MODEL FORMULA (per lap):
  lap_time = base_lap + delta[compound] + deg[compound] * (tire_age ^ curve) * temp_factor
  temp_factor = 1.0 + (track_temp - 30.0) * t_coeff

PARAMS (7 unknowns):
  [M_delta, H_delta, S_deg, M_deg, H_deg, t_coeff, curve]
  SOFT delta is fixed at 0.0 (reference / fastest compound)

REGULATIONS COMPLIANCE:
  - tire_age increments BEFORE lap time calculation (starts at 1 on first lap)
  - Pit stop penalty added at END of the lap
  - tire_age resets to 0 on pit, becomes 1 on next lap
  - base_lap + pit penalties are known constants; only deltas/degs affect ordering

Usage:
  python crack_constants.py [n_races]
  python crack_constants.py 1000   # use 1000 training races (more = more accurate)

AWS recommended: c6i.32xlarge (128 vCPUs) — converges in ~3-8 minutes
"""

import sys
import json
import glob
import random
import os
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TIRE_MAP   = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
TIRE_NAMES = ["SOFT", "MEDIUM", "HARD"]
TEMP_REF   = 30.0   # Reference temperature for normalisation (degrees C)
RANDOM_SEED = 42

# Bounds for the 7 parameters
# SOFT is FASTEST (delta=0 reference), HARD is SLOWEST (+highest delta)
# SOFT DEGRADES FASTEST, HARD degrades SLOWEST
PARAM_BOUNDS = [
    (0.0,   8.0),   # M_delta:  MEDIUM is slower than SOFT
    (0.0,  16.0),   # H_delta:  HARD is slower than MEDIUM (and SOFT)
    (0.0,   3.0),   # S_deg:    SOFT degrades fastest
    (0.0,   2.0),   # M_deg:    MEDIUM degrades moderately
    (0.0,   1.5),   # H_deg:    HARD degrades slowest
    (-0.05, 0.05),  # t_coeff:  temp effect on degradation (+ve = hotter = more deg)
    (0.5,   3.0),   # curve:    power-law exponent for degradation shape
]
PARAM_NAMES = ["M_DELTA", "H_DELTA", "S_DEG", "M_DEG", "H_DEG", "T_COEFF", "CURVE"]


# ─────────────────────────────────────────────
# PREPROCESSING  (done once, before optimisation)
# ─────────────────────────────────────────────

def preprocess_race(race: dict) -> dict:
    """
    Convert a raw race record into a compact numerical representation.
    
    Returns:
        {
          'finishing_positions': [driver_id, ...],  # ground truth order
          'drivers': {
              driver_id: {
                  'base_time': float,   # fixed component (base laps + pit penalties)
                  'temp':      int,     # track temperature
                  'stints':    [(compound_idx, n_laps), ...]  # tire stints
              }
          }
        }
    """
    cfg        = race['race_config']
    total_laps = cfg['total_laps']
    base_lap   = cfg['base_lap_time']
    pit_time   = cfg['pit_lane_time']
    temp       = cfg['track_temp']

    drivers = {}
    for _, strat in race['strategies'].items():
        driver_id    = strat['driver_id']
        pits_sorted  = sorted(strat['pit_stops'], key=lambda x: x['lap'])

        stints = []
        prev_lap      = 0
        curr_compound = TIRE_MAP[strat['starting_tire']]

        for pit in pits_sorted:
            pit_lap    = pit['lap']
            stint_laps = pit_lap - prev_lap          # laps on this compound
            if stint_laps > 0:
                stints.append((curr_compound, stint_laps))
            curr_compound = TIRE_MAP[pit['to_tire']]
            prev_lap      = pit_lap

        # Last stint to end of race
        final_laps = total_laps - prev_lap
        if final_laps > 0:
            stints.append((curr_compound, final_laps))

        n_pits    = len(pits_sorted)
        base_time = total_laps * base_lap + n_pits * pit_time  # constant regardless of params

        drivers[driver_id] = {
            'base_time': base_time,
            'temp':      temp,
            'stints':    stints,
        }

    return {
        'finishing_positions': race['finishing_positions'],
        'drivers':             drivers,
    }


# ─────────────────────────────────────────────
# SIMULATION  (vectorised per driver)
# ─────────────────────────────────────────────

def driver_total_time(driver_data: dict, params: np.ndarray) -> float:
    """
    Compute a single driver's total race time given the parameter vector.

    Formula (per lap):
        lap_time = base_lap + delta[c] + deg[c] * (tire_age ^ curve) * temp_factor

    Vectorised over laps within each stint:
        stint_contribution = n_laps * delta[c]
                           + deg[c] * temp_factor * sum(1^curve + 2^curve + ... + n^curve)
    """
    m_d, h_d, s_deg, m_deg, h_deg, t_coeff, curve = params
    deltas = (0.0, m_d, h_d)                          # SOFT=0 reference
    degs   = (s_deg, m_deg, h_deg)

    temp_factor = 1.0 + (driver_data['temp'] - TEMP_REF) * t_coeff
    total       = driver_data['base_time']

    for (compound_idx, n_laps) in driver_data['stints']:
        # ages = [1, 2, ..., n_laps]  (REGULATION: age starts at 1)
        ages    = np.arange(1, n_laps + 1, dtype=np.float64)
        age_sum = np.sum(ages ** curve)

        total += n_laps * deltas[compound_idx] + degs[compound_idx] * temp_factor * age_sum

    return total


# ─────────────────────────────────────────────
# OBJECTIVE FUNCTION
# ─────────────────────────────────────────────

def ranking_loss_for_race(params: np.ndarray, race: dict) -> float:
    """
    Full pairwise Kendall-tau ranking loss for one race.
    
    Penalises EVERY pair (winner, loser) where the predicted time
    order contradicts the actual finishing order.
    
    Using quadratic hinge loss: max(0, margin + epsilon)^2
    This gives smooth gradients and penalises large violations more.
    
    ADVANTAGE over adjacent-pair check:
      If D001 should finish 1st but we predict 20th, adjacent pairs
      only see 1 violation. Full pairwise sees 19 violations — much
      stronger gradient signal pulling the params toward the truth.
    """
    drivers_data   = race['drivers']
    actual_order   = race['finishing_positions']

    times = {did: driver_total_time(drivers_data[did], params)
             for did in actual_order}

    loss = 0.0
    n    = len(actual_order)

    for i in range(n - 1):
        t_winner = times[actual_order[i]]       # should be LOWER (faster)
        for j in range(i + 1, n):
            t_loser = times[actual_order[j]]    # should be HIGHER (slower)
            margin  = t_winner - t_loser        # correct ordering → margin < 0
            if margin >= 0:                     # violation: winner is NOT faster
                loss += (margin + 0.1) ** 2     # +0.1 epsilon ensures gradient even at margin=0

    return loss


# Global races list — required because multiprocessing workers can't receive large args
_RACES: list = []

def _objective(params: np.ndarray) -> float:
    """Total ranking loss across all training races."""
    return sum(ranking_loss_for_race(params, r) for r in _RACES)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_diverse_races(n_races: int = 500) -> list:
    """
    Load a diverse sample of races, spread evenly across all data files.
    Diversity across tracks / temperatures helps the optimizer generalise.
    """
    files = sorted(glob.glob('data/historical_races/*.json'))
    if not files:
        print("ERROR: No historical race data found at data/historical_races/*.json")
        print("Make sure you're running from the repository root.")
        sys.exit(1)

    rng              = random.Random(RANDOM_SEED)
    target_per_file  = max(1, n_races // len(files) + 1)
    preprocessed     = []

    for fpath in files:
        with open(fpath, 'r') as f:
            races = json.load(f)
        sample = rng.sample(races, min(target_per_file, len(races)))
        preprocessed.extend(preprocess_race(r) for r in sample)
        if len(preprocessed) >= n_races:
            break

    rng.shuffle(preprocessed)
    result = preprocessed[:n_races]

    # Print diversity stats
    temps  = [list(r['drivers'].values())[0]['temp'] for r in result]
    print(f"  Temperature range: {min(temps)}–{max(temps)} °C")

    return result


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def evaluate_accuracy(params: np.ndarray, races: list) -> float:
    """Compute exact ordering accuracy (0–100%)."""
    correct = 0
    for race in races:
        drivers_data = race['drivers']
        times        = {did: driver_total_time(drivers_data[did], params)
                        for did in drivers_data}
        predicted    = sorted(times.keys(), key=lambda d: times[d])
        if predicted == race['finishing_positions']:
            correct += 1
    return correct / len(races) * 100.0


# ─────────────────────────────────────────────
# MAIN CRACKER
# ─────────────────────────────────────────────

def crack_constants(n_races: int = 500):
    global _RACES

    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1

    print("=" * 60)
    print("  F1 Constants Cracker — Box Box Box Challenge")
    print("=" * 60)
    print(f"  CPUs available    : {cpu_count}")
    print(f"  Training races    : {n_races}")
    print(f"  Parameters        : {len(PARAM_BOUNDS)} unknowns")
    print()

    # ── Load data ──────────────────────────────
    print(f"Loading {n_races} diverse races...", flush=True)
    t0     = time.time()
    _RACES = load_diverse_races(n_races)
    print(f"  Loaded {len(_RACES)} races in {time.time()-t0:.1f}s\n", flush=True)

    # ── Phase 1: Differential Evolution ────────
    print("Phase 1: Global Search — Differential Evolution")
    print(f"  workers=-1 → {cpu_count} parallel CPU workers")
    print(f"  population: 15 × {len(PARAM_BOUNDS)} = {15*len(PARAM_BOUNDS)} individuals")
    print("  (This will print loss per generation below)\n", flush=True)

    t1 = time.time()
    result_de = differential_evolution(
        _objective,
        PARAM_BOUNDS,
        maxiter     = 2000,
        popsize     = 15,          # 15 * n_params individuals per generation
        tol         = 1e-14,       # convergence tolerance
        mutation    = (0.5, 1.5),  # F mutation factor range
        recombination = 0.9,       # crossover probability
        seed        = RANDOM_SEED,
        workers     = -1,          # ← USE ALL CPU CORES (multiprocessing.Pool)
        updating    = 'deferred',  # required for workers != 1
        init        = 'latinhypercube',   # better coverage than 'random'
        disp        = True,        # print each generation
        polish      = False,       # we'll polish with L-BFGS-B for higher precision
    )
    print(f"\n  DE completed in {time.time()-t1:.1f}s")
    print(f"  DE converged: {result_de.success}")
    print(f"  DE final loss: {result_de.fun:.8f}\n", flush=True)

    # ── Phase 2: L-BFGS-B precision polish ─────
    print("Phase 2: Precision Polish — L-BFGS-B", flush=True)
    t2 = time.time()
    result_final = minimize(
        _objective,
        result_de.x,
        method  = 'L-BFGS-B',
        bounds  = PARAM_BOUNDS,
        options = {
            'ftol':    1e-20,
            'gtol':    1e-16,
            'maxiter': 10000,
            'maxfun':  100000,
        }
    )
    print(f"  L-BFGS-B completed in {time.time()-t2:.1f}s")
    print(f"  Final loss: {result_final.fun:.12f}\n", flush=True)

    # ── Results ─────────────────────────────────
    params = result_final.x
    print("=" * 60)
    print("  CRACKED CONSTANTS")
    print("=" * 60)
    for name, val in zip(PARAM_NAMES, params):
        print(f"  {name:<12} = {val:.15f}")
    print()

    # Physical sanity checks
    print("Sanity Checks:")
    m_d, h_d, s_deg, m_deg, h_deg, t_coeff, curve = params
    print(f"  SOFT faster than MEDIUM: delta_M={m_d:.4f} > 0  → {'✓' if m_d > 0 else '✗ WARNING'}")
    print(f"  MEDIUM faster than HARD: delta_H={h_d:.4f} > delta_M → {'✓' if h_d > m_d else '✗ WARNING'}")
    print(f"  SOFT degrades fastest:   S_deg={s_deg:.4f} > M_deg={m_deg:.4f} → {'✓' if s_deg > m_deg else '✗ WARNING'}")
    print(f"  MEDIUM > HARD deg rate:  M_deg={m_deg:.4f} > H_deg={h_deg:.4f} → {'✓' if m_deg > h_deg else '✗ WARNING'}")
    print(f"  Curve exponent:          {curve:.4f}  (expected 1–2)")
    print()

    # Accuracy on training data
    print("Evaluating training accuracy...", flush=True)
    accuracy = evaluate_accuracy(params, _RACES)
    print(f"  Training accuracy: {accuracy:.1f}%")

    if accuracy < 80:
        print("  ⚠  Below 80% — try increasing n_races or check data path")
    elif accuracy < 95:
        print("  ✓  Good result — may improve further with more races")
    else:
        print("  ✓✓ Excellent! Constants appear well-converged")
    print()

    # ── Save constants ───────────────────────────
    os.makedirs('solution', exist_ok=True)
    constants = dict(zip(PARAM_NAMES, params.tolist()))
    constants['_training_accuracy_pct'] = accuracy
    constants['_n_training_races']      = len(_RACES)
    constants['_final_loss']            = float(result_final.fun)

    output_path = 'solution/constants.json'
    with open(output_path, 'w') as f:
        json.dump(constants, f, indent=2)
    print(f"Constants saved to: {output_path}")
    print(f"Total time: {time.time()-t0:.1f}s")

    return params


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    crack_constants(n)
