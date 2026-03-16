#!/usr/bin/env python3
"""
F1 Race Simulator — Box Box Box Challenge
==========================================
Reads race JSON from stdin, outputs finishing positions to stdout.

Usage:
    cat data/test_cases/inputs/test_001.json | python solution/race_simulator.py

Requirements:
    - Run crack_constants.py first to generate solution/constants.json
    - Python 3.8+, numpy

Formula (per lap):
    lap_time = base_lap
             + delta[compound]                              ← compound speed offset
             + deg[compound] * (tire_age^curve)             ← degradation (power law)
             * (1 + (track_temp - 30) * t_coeff)           ← temperature scaling

REGULATIONS COMPLIANCE (from regulations.md):
    - tire_age starts at 0 when fitted; increments to 1 BEFORE the first lap is driven
    - Pit stop time penalty applied at END of the pit lap
    - All 20 cars simulate independently (no car interactions)
    - Finishing order = ascending total race time
"""

import sys
import json
import os
import numpy as np

# ─────────────────────────────────────────────
# TIRE MAP
# ─────────────────────────────────────────────
TIRE_MAP   = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
TEMP_REF   = 30.0

# ─────────────────────────────────────────────
# LOAD CONSTANTS
# ─────────────────────────────────────────────

def load_constants() -> tuple:
    """
    Load the 7 cracked constants from solution/constants.json.
    Run crack_constants.py first to generate this file.
    
    Returns:
        (M_delta, H_delta, S_deg, M_deg, H_deg, t_coeff, curve)
    """
    # Look for constants.json relative to this script's directory or repo root
    candidates = [
        os.path.join(os.path.dirname(__file__), 'constants.json'),
        'solution/constants.json',
        'constants.json',
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r') as f:
                c = json.load(f)
            return (
                c['M_DELTA'], c['H_DELTA'],
                c['S_DEG'],   c['M_DEG'],   c['H_DEG'],
                c['T_COEFF'], c['CURVE'],
            )

    # If no constants file found, fail with a clear message
    raise FileNotFoundError(
        "solution/constants.json not found!\n"
        "Run crack_constants.py first:\n"
        "  python crack_constants.py 500\n"
    )


# ─────────────────────────────────────────────
# CORE SIMULATION
# ─────────────────────────────────────────────

def simulate_race(race_data: dict, params: tuple = None) -> dict:
    """
    Simulate a full race and return finishing positions.

    Args:
        race_data: Full race JSON (race_config + strategies)
        params:    Optional 7-tuple of constants. Loads from file if None.

    Returns:
        {"race_id": str, "finishing_positions": [driver_id, ...]}
    """
    if params is None:
        params = load_constants()

    m_d, h_d, s_deg, m_deg, h_deg, t_coeff, curve = params

    # Lookup arrays indexed by TIRE_MAP
    deltas = np.array([0.0,   m_d,   h_d],   dtype=np.float64)  # SOFT=0 reference
    degs   = np.array([s_deg, m_deg, h_deg], dtype=np.float64)

    cfg        = race_data['race_config']
    base_lap   = float(cfg['base_lap_time'])
    track_temp = float(cfg['track_temp'])
    pit_time   = float(cfg['pit_lane_time'])
    total_laps = int(cfg['total_laps'])

    # Temperature multiplier on degradation
    temp_factor = 1.0 + (track_temp - TEMP_REF) * t_coeff

    driver_times = {}

    for _, strat in race_data['strategies'].items():
        driver_id  = strat['driver_id']
        pits_sorted = sorted(strat['pit_stops'], key=lambda x: x['lap'])

        # Build pit-lap → new_compound lookup for O(1) per-lap access
        pit_dict = {int(p['lap']): TIRE_MAP[p['to_tire']] for p in pits_sorted}

        total_time   = 0.0
        curr_compound = TIRE_MAP[strat['starting_tire']]
        tire_age      = 0   # REGULATION: starts at 0, increments to 1 before first lap

        for lap in range(1, total_laps + 1):
            # REGULATION: age increments BEFORE lap time is calculated
            tire_age += 1

            total_time += (
                base_lap
                + deltas[curr_compound]
                + degs[curr_compound] * (tire_age ** curve) * temp_factor
            )

            # REGULATION: pit stop penalty at END of the lap
            if lap in pit_dict:
                total_time    += pit_time
                curr_compound  = pit_dict[lap]
                tire_age       = 0   # reset: new tires fitted

        driver_times[driver_id] = total_time

    # Sort by ascending total time → 1st to 20th
    finishing_order = sorted(driver_times.keys(), key=lambda d: driver_times[d])

    return {
        'race_id':            race_data['race_id'],
        'finishing_positions': finishing_order,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    race_data = json.load(sys.stdin)
    result    = simulate_race(race_data)
    print(json.dumps(result))
