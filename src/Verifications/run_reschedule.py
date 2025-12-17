#!/usr/bin/env python3
# -- coding: utf-8 --

import json
from solver_core import make_base_data, solve_reschedule

def main():
    # load baseline
    with open("baseline_solution.json", "r", encoding="utf-8") as f:
        old_solution = json.load(f)

    data = make_base_data()

    # ---- inputs (later UI will send these) ----
    mode = input("RUNNING ops: [1] devam  [2] model karar versin (default=1): ").strip()
    mode = "optimize" if mode == "2" else "continue"

    unavail_m = input("Unavailable machines (e.g. 4,5) or empty: ").strip()
    unavail_l = input("Unavailable stations (e.g. 2,9) or empty: ").strip()

    def parse_list(s):
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    unavailable_machines = parse_list(unavail_m)
    unavailable_stations = parse_list(unavail_l)

    # urgent config (UI’den gelecek)
    urgent_ops_count = 3
    urgent_due_slack = 10.0

    res = solve_reschedule(
        data_base=data,
        old_solution=old_solution,
        shift_start_hhmm="08:00",
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        mode=mode,
        urgent_job_id=9,
        urgent_ops_count=urgent_ops_count,
        urgent_due_slack=urgent_due_slack
    )

    with open("reschedule_solution.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("\n✅ Reschedule solved and saved to reschedule_solution.json")
    print("t0 =", res["t0"])
    print("Objective:", res["objective"])
    print("Urgent:", res["urgent"])
    if res["keep_decisions"]:
        print("keep decisions:", res["keep_decisions"])

if __name__ == "__main__":
    main()
