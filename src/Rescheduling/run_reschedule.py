#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_reschedule.py

BASELINE:
  - HeuristicBaseModel.run_heuristic(data) -> baseline_solution.json (senin eski formatında)

RESCHEDULE:
  - solver_core.solve_reschedule_open_source(...) -> reschedule_solution.json

SCENARIO COMPAT:
  - supports:
      plan_start_iso, plan_calendar, scenario_name
      baseline_overrides, overrides
      disruptions { unavailable_machines, unavailable_stations }
      urgent_job / urgent_payload / urgent_payload_path
      reschedule { ... same keys ... }
  - if scenario has top-level "data" dict, uses it
    else uses solver_core.make_base_data(overrides=baseline_overrides)
"""

import json
import os
import sys
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Tuple, Optional

from HeuristicBaseModel import run_heuristic  # ✅ baseline burada

# ✅ Open-source rescheduling + base data builder burada olmalı
from solver_core import make_base_data, solve_reschedule_open_source


# ==========================================================
#  JSON IO
# ==========================================================

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ==========================================================
#  HELPERS
# ==========================================================

def _tuple_key_str(i: int, k: int) -> str:
    return f"{int(i)},{int(k)}"

def _scenario_tag(scn: dict, scenario_path: str) -> str:
    sid = scn.get("scenario_id", None)
    if sid is not None:
        s = str(sid).strip()
        return f"s{s.zfill(2)}" if s.isdigit() else f"s_{s}"

    base = os.path.basename(scenario_path).replace(".json", "")
    parts = base.split("_")
    for p in parts:
        if p.isdigit():
            return f"s{p.zfill(2)}"

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"s_{ts}"

def _pick_scenario_name(scn: dict, scenario_path: str) -> str:
    return (
        scn.get("scenario_name")
        or scn.get("name")
        or scn.get("title")
        or os.path.basename(scenario_path)
    )

def _pick_outputs(scn: dict, workdir: str):
    base_out = scn.get("baseline_output") or "baseline_solution.json"
    res_out = scn.get("reschedule_output") or "reschedule_solution.json"

    if not os.path.isabs(base_out):
        base_out = os.path.join(workdir, base_out)
    if not os.path.isabs(res_out):
        res_out = os.path.join(workdir, res_out)

    canonical_base = os.path.join(workdir, "baseline_solution.json")
    canonical_res = os.path.join(workdir, "reschedule_solution.json")
    return base_out, res_out, canonical_base, canonical_res

def _run_plotter_and_archive(tag: str, workdir: str):
    plot_script = os.path.join(workdir, "plot_gantt_all.py")
    if not os.path.exists(plot_script):
        print(f"⚠️ plot_gantt_all.py not found at: {plot_script}")
        return

    print("📊 Generating Gantt charts (with legend)...")
    subprocess.run([sys.executable, plot_script], cwd=workdir, check=True)

    outputs = [
        "gantt_machine_baseline.png",
        "gantt_station_baseline.png",
        "gantt_machine_reschedule.png",
        "gantt_station_reschedule.png",
    ]

    for fn in outputs:
        src = os.path.join(workdir, fn)
        if os.path.exists(src):
            name, ext = os.path.splitext(fn)
            dst = os.path.join(workdir, f"{name}_{tag}{ext}")
            shutil.copyfile(src, dst)
            print(f"✅ Saved: {dst}")

def _apply_overrides_to_data(data: dict, overrides: dict) -> dict:
    """
    make_base_data(overrides=...) ile aynı tarz merge:
    - dict ise key bazında override eder
    - diğer tiplerde direkt replace
    """
    if not overrides:
        return data

    out = dict(data)
    for key, val in overrides.items():
        if key not in out:
            raise ValueError(f"Unknown override key in scenario: {key}")

        if isinstance(out[key], dict) and isinstance(val, dict):
            new_map = dict(out[key])
            for kk, vv in val.items():
                try:
                    ik = int(kk)
                except Exception:
                    ik = kk
                new_map[ik] = vv
            out[key] = new_map
        else:
            out[key] = val

    return out

def _pick_disruptions(scn: dict):
    if "disruptions" in scn and isinstance(scn["disruptions"], dict):
        d = scn["disruptions"]
        return d.get("unavailable_machines", []), d.get("unavailable_stations", [])

    if "reschedule" in scn and isinstance(scn["reschedule"], dict):
        r = scn["reschedule"]
        if "disruptions" in r and isinstance(r["disruptions"], dict):
            d = r["disruptions"]
            return d.get("unavailable_machines", []), d.get("unavailable_stations", [])
        return r.get("unavailable_machines", []), r.get("unavailable_stations", [])

    return scn.get("unavailable_machines", []), scn.get("unavailable_stations", [])

def _pick_mode(scn: dict) -> str:
    # tercih 1 => continue
    if "mode" in scn:
        return scn.get("mode") or "continue"
    if "reschedule" in scn and isinstance(scn["reschedule"], dict):
        return scn["reschedule"].get("mode", "continue")
    return "continue"

def _load_urgent_payload_from_path(path: str, workdir: str) -> dict:
    if not path:
        return {}

    # relative ise workdir'e göre dene
    if not os.path.isabs(path):
        p2 = os.path.join(workdir, path)
        if os.path.exists(p2):
            path = p2

    if not os.path.exists(path):
        raise FileNotFoundError(f"urgent_payload_path not found: {path}")

    uj = _load_json(path)

    # desteklenen formatlar:
    # { "urgent_job": {...} }  veya { "job_id": ... }
    if isinstance(uj, dict) and "urgent_job" in uj:
        return uj
    if isinstance(uj, dict) and "job_id" in uj:
        return {"urgent_job": uj}
    return uj if isinstance(uj, dict) else {}

def _pick_urgent_payload(scn: dict, workdir: str) -> dict:
    """
    Supports multiple formats:
    - "urgent_job": {...}
    - "urgent_payload": {...}
    - "urgent_payload_path": "scenarios/urgent_x.json"
    - nested: "reschedule": {"urgent_payload_path": "..."} / {"urgent_job":...} / {"urgent_payload":...}
    """
    if "urgent_job" in scn and isinstance(scn["urgent_job"], dict):
        return {"urgent_job": scn["urgent_job"]}

    if "urgent_payload" in scn and isinstance(scn["urgent_payload"], dict):
        return scn["urgent_payload"]

    if "urgent_payload_path" in scn:
        return _load_urgent_payload_from_path(scn.get("urgent_payload_path"), workdir)

    if "reschedule" in scn and isinstance(scn["reschedule"], dict):
        rp = scn["reschedule"]
        if "urgent_job" in rp and isinstance(rp["urgent_job"], dict):
            return {"urgent_job": rp["urgent_job"]}
        if "urgent_payload" in rp and isinstance(rp["urgent_payload"], dict):
            return rp["urgent_payload"]
        if "urgent_payload_path" in rp:
            return _load_urgent_payload_from_path(rp.get("urgent_payload_path"), workdir)

    return {}

def _adapt_data_for_heuristic(data: dict) -> dict:
    """
    HeuristicBaseModel verify_data_basic beklediği anahtarlar için adapter:
      - p_flag_j bekliyor; bizde p_j varsa kopyala
    """
    d = dict(data)
    if "p_flag_j" not in d and "p_j" in d:
        d["p_flag_j"] = d["p_j"]
    return d


def _baseline_to_solution_json(
    plan_start_iso: str,
    plan_calendar: dict,
    scenario_name: str,
    data: dict,
    heuristic_res
) -> dict:
    """
    HeuristicBaseModel.run_heuristic çıktısını senin eski baseline JSON formatına çevirir.
    """
    I = [int(x) for x in data["I"]]

    # S_old / C_old
    S_old = {int(i): float(heuristic_res.S[int(i)]) for i in I}
    C_old = {int(i): float(heuristic_res.C[int(i)]) for i in I}

    # x_old / y_old (string key format "i,k")
    x_old = {}
    y_old = {}
    for i in I:
        m = int(heuristic_res.assign_machine[i])
        l = int(heuristic_res.assign_station[i])
        x_old[_tuple_key_str(i, m)] = 1
        y_old[_tuple_key_str(i, l)] = 1

    # job_of
    job_of = {}
    for j, ops in data["O_j"].items():
        for op in ops:
            job_of[int(op)] = int(j)

    schedule = []
    for i in I:
        schedule.append({
            "op_id": int(i),
            "op_label": str(int(i)),
            "job_id": int(job_of.get(int(i), -1)),
            "start": float(heuristic_res.S[int(i)]),
            "finish": float(heuristic_res.C[int(i)]),
            "machine": int(heuristic_res.assign_machine[int(i)]),
            "station": int(heuristic_res.assign_station[int(i)]),
        })

    schedule.sort(key=lambda r: (float(r["start"]), float(r["finish"]), int(r["op_id"])))

    return {
        "plan_start_iso": plan_start_iso,
        "plan_calendar": plan_calendar,
        "scenario_name": scenario_name,
        "objective": {"T_max": float(heuristic_res.T_max), "C_max": float(heuristic_res.C_max)},
        "schedule": schedule,
        "S_old": S_old,
        "C_old": C_old,
        "x_old": x_old,
        "y_old": y_old,
    }


# ==========================================================
#  MAIN
# ==========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_reschedule_open_source.py scenarios/<scenario>.json")
        sys.exit(1)

    scenario_path = sys.argv[1]
    scn = _load_json(scenario_path)
    workdir = os.path.dirname(os.path.abspath(__file__))

    scenario_name = _pick_scenario_name(scn, scenario_path)
    plan_start_iso = scn.get("plan_start_iso", "2025-12-18T05:00:00+00:00")
    plan_calendar = scn.get("plan_calendar", {"utc_offset": "+03:00"})

    # Baseline overrides (data üretirken)
    baseline_overrides = scn.get("baseline_overrides", {}) or {}
    # Reschedule overrides (t0 sonrası etkili kabul edip data’ya reschedule aşamasında uyguluyoruz)
    reschedule_overrides = scn.get("overrides", {}) or {}

    # -------------------------
    # DATA SOURCE
    # -------------------------
    # 1) Eğer senaryo JSON top-level "data" içeriyorsa onu kullan
    # 2) Yoksa make_base_data(overrides=baseline_overrides)
    if "data" in scn and isinstance(scn["data"], dict):
        data_baseline = scn["data"]
        # baseline_overrides varsa bunun üstüne uygula (opsiyonel ama tutarlı)
        if baseline_overrides:
            data_baseline = _apply_overrides_to_data(data_baseline, baseline_overrides)
    else:
        data_baseline = make_base_data(overrides=baseline_overrides)

    tag = _scenario_tag(scn, scenario_path)
    baseline_out, reschedule_out, canonical_base, canonical_res = _pick_outputs(scn, workdir)

    # ==========================================================
    # BASELINE (HeuristicBaseModel)
    # ==========================================================
    data_for_heur = _adapt_data_for_heuristic(data_baseline)
    k1 = float(scn.get("k1", 2.0))
    base_res = run_heuristic(data_for_heur, k1=k1)

    baseline = _baseline_to_solution_json(
        plan_start_iso=plan_start_iso,
        plan_calendar=plan_calendar,
        scenario_name=scenario_name,
        data=data_for_heur,  # heuristic hangi data ile çalıştıysa onu baz al
        heuristic_res=base_res
    )

    _save_json(baseline_out, baseline)
    _save_json(canonical_base, baseline)

    print(f"✅ Baseline saved: {baseline_out}")
    print("scenario:", scenario_name)
    print("plan_start_iso (UTC):", baseline["plan_start_iso"])
    print("objective:", baseline["objective"])

    # ==========================================================
    # RESCHEDULE DATA (baseline + reschedule overrides)
    # ==========================================================
    data_reschedule = _apply_overrides_to_data(data_baseline, reschedule_overrides)
    # reschedule tarafında da p_flag_j uyumu gerekebilir
    data_reschedule = _adapt_data_for_heuristic(data_reschedule)

    # ==========================================================
    # RESCHEDULE (Open-source heuristic)
    # ==========================================================
    unavailable_machines, unavailable_stations = _pick_disruptions(scn)
    mode = _pick_mode(scn)  # tercih 1 => continue

    urgent_payload = _pick_urgent_payload(scn, workdir)
    if not urgent_payload:
        urgent_payload = None

    res = solve_reschedule_open_source(
        data_base=data_reschedule,
        old_solution=baseline,
        urgent_payload=urgent_payload,
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        mode=mode,
    )

    res["scenario_name"] = scenario_name
    res["mode"] = mode
    res["disruptions"] = {
        "unavailable_machines": unavailable_machines,
        "unavailable_stations": unavailable_stations
    }
    res["applied_overrides_at_reschedule"] = reschedule_overrides

    _save_json(reschedule_out, res)
    _save_json(canonical_res, res)

    print(f"✅ Reschedule saved: {reschedule_out}")
    print("t0:", res.get("t0"))
    print("objective:", res.get("objective"))

    _run_plotter_and_archive(tag=tag, workdir=workdir)


if __name__ == "__main__":
    main()
