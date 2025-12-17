#!/usr/bin/env python3
# -- coding: utf-8 --

import json
from solver_core import make_base_data, solve_baseline

def main():
    data = make_base_data()
    sol = solve_baseline(data)

    with open("baseline_solution.json", "w", encoding="utf-8") as f:
        json.dump(sol, f, ensure_ascii=False, indent=2)

    print("✅ Baseline solved and saved to baseline_solution.json")
    print("Objective:", sol["objective"])

if __name__ == "__main__":
    main()
