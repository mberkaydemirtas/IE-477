#!/usr/bin/env python3
# -- coding: utf-8 --

import json
from pathlib import Path

from Rescheduling.solver_core import (
    make_base_data,
    normalize_old_solution,
    compute_t0_from_plan_start,
    classify_ops_by_t0,
    add_urgent_job_from_payload,
    add_split_remainders_for_running_ops,
)

def load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    base_path = Path(".")  # Verifications klasörü içinden çalıştırıyorsun varsayıyorum

    baseline = load_json(str(base_path / "baseline_solution.json"))
    baseline = normalize_old_solution(baseline)

    urgent_payload = load_json(str(base_path / "urgent_job.json"))

    data_base = make_base_data()

    # t0 (working-hours)
    plan_calendar = baseline.get("plan_calendar", {})
    t0 = compute_t0_from_plan_start(baseline["plan_start_iso"], plan_calendar=plan_calendar)

    I = data_base["I"]
    I_done, I_run, I_free = classify_ops_by_t0(I, baseline["S_old"], baseline["C_old"], t0)

    print("\n=== t0 & sets ===")
    print("t0 =", t0)
    print("I_run =", I_run)

    # data2: urgent + split remainders
    data1 = add_urgent_job_from_payload(data_base, t0=t0, urgent_payload=urgent_payload)
    data2 = add_split_remainders_for_running_ops(data1, I_run, t0, baseline)

    Pred_i = data2["Pred_i"]
    rem_map = data2.get("rem_map", {})

    print("\n=== rem_map (running -> remainder) ===")
    for k, v in rem_map.items():
        print(f"{k} -> {v}")

    # Özellikle beklediğimiz: 15->104 ve 22->105 (senin çıktında böyleydi)
    targets = []
    for run_op in I_run:
        if run_op in rem_map:
            targets.append((run_op, rem_map[run_op]))

    print("\n=== CHECK 1: remainder op predecessors should include the running op ===")
    for run_op, rem_op in targets:
        preds = Pred_i.get(rem_op, [])
        ok = (run_op in preds)
        print(f"rem_op {rem_op} preds = {preds}  ==> contains {run_op}? {ok}")

    print("\n=== CHECK 2: successors of running op should also list remainder as predecessor ===")
    # “successor” = Pred_i[k] içinde run_op geçen tüm k’lar
    for run_op, rem_op in targets:
        successors = [k for k, preds in Pred_i.items() if run_op in preds and k != rem_op]
        print(f"\nRunning op {run_op} successors (excluding remainder itself): {successors}")

        bad = []
        for k in successors:
            if rem_op not in Pred_i.get(k, []):
                bad.append(k)

        if not bad:
            print(f"OK: All successors of {run_op} include remainder {rem_op} as predecessor.")
        else:
            print(f"❌ PROBLEM: These successors of {run_op} DO NOT include remainder {rem_op}: {bad}")

    # İstersen urgent ops için de pred’leri yazdıralım
    urgent_ops = data2.get("urgent_ops", [])
    if urgent_ops:
        print("\n=== Urgent ops Pred_i ===")
        for u in urgent_ops:
            print(f"{u} preds = {Pred_i.get(u, [])}")

    print("\n✅ Verification finished.")

if __name__ == "__main__":
    main()

