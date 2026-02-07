# -*- coding: utf-8 -*-
"""
adapter.py

Converts incoming ERP-like OPERATION list into the solver "main json format".

Key rules:
- Jobs are grouped by parentId (or workOrderNumber fallback)
- Pred_i is built ONLY within each job using ordering:
    erpOperationItemNumber (numeric) -> plannedStartDateTime -> id
  and then chain precedence: op(k) -> op(k+1)
- r_j from min plannedStartDateTime of job (hours from plan_start)
- d_j from max plannedEndDateTime of job (hours from plan_start)
- p_i = cycleTime/60 hours, if null -> defaults.default_processing_time_hours
- Machines/stations:
    machineCandidates if exists else ALL machines (if allow_all_machines_if_missing)
    stationCandidates if exists else ALL stations (if allow_all_stations_if_missing)
- Station size:
    stationSizeRequirement = "BIG"/"SMALL"/"ANY"
    -> beta_i=1 only if BIG required (i.e., cannot use SMALL stations)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone
import math


def _parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    # allow 'Z'
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        # assume UTC if missing timezone
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hours_between(a: datetime, b: datetime) -> float:
    # b - a in hours
    return (b - a).total_seconds() / 3600.0


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _as_int_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for t in x:
            v = _safe_int(t, None)
            if v is not None:
                out.append(v)
        return out
    return []


def build_data_from_operations(
    operations: List[dict],
    base_meta: Dict[str, Any],
    plan_start_iso: str,
    plan_calendar: Optional[dict],
    system_config: Dict[str, Any],
) -> Dict[str, Any]:
    plan_calendar = plan_calendar or system_config.get("calendar", {}) or {"utc_offset": "+03:00"}
    plan_start_dt = _parse_iso_dt(plan_start_iso)
    if plan_start_dt is None:
        raise ValueError("plan_start_iso is invalid")

    defaults = (system_config.get("defaults") or {})
    default_pt = float(defaults.get("default_processing_time_hours", 1.0))
    allow_all_m = bool(defaults.get("allow_all_machines_if_missing", True))
    allow_all_l = bool(defaults.get("allow_all_stations_if_missing", True))

    machines = system_config.get("machines", {}) or {}
    stations = system_config.get("stations", {}) or {}
    M = _as_int_list(machines.get("machine_ids"))
    L = _as_int_list(stations.get("station_ids"))

    station_types = stations.get("station_types", {}) or {}
    L_big = [int(k) for k, v in station_types.items() if str(v).upper() == "BIG"]
    L_small = [int(k) for k, v in station_types.items() if str(v).upper() == "SMALL"]
    if not L_big and not L_small:
        # fallback: treat none as small list empty
        L_big = []
        L_small = []

    # -------------------------
    # Group operations into jobs
    # -------------------------
    # Prefer parentId, fallback to workOrderNumber.
    groups: Dict[str, List[dict]] = {}
    for op in operations:
        if not isinstance(op, dict):
            continue
        parent_id = op.get("parentId", None)
        if parent_id is not None:
            key = f"PID:{parent_id}"
        else:
            wo = op.get("workOrderNumber", None) or op.get("parentName", None) or "UNKNOWN"
            key = f"WO:{wo}"
        groups.setdefault(key, []).append(op)

    # Assign job IDs 1..n deterministically
    group_keys = sorted(groups.keys())
    job_id_of_group: Dict[str, int] = {gk: idx + 1 for idx, gk in enumerate(group_keys)}

    J = [job_id_of_group[gk] for gk in group_keys]

    # -------------------------
    # Build I and O_j with ordering
    # -------------------------
    def _op_sort_key(op: dict) -> Tuple[int, float, int]:
        # erpOperationItemNumber numeric
        item = op.get("erpOperationItemNumber", None)
        item_int = _safe_int(item, 10**9)
        # plannedStartDateTime
        ps = _parse_iso_dt(op.get("plannedStartDateTime"))
        ps_h = _hours_between(plan_start_dt, ps) if ps else 10**9
        # id
        oid = _safe_int(op.get("id"), 10**9)
        return (item_int, ps_h, oid)

    I: List[int] = []
    O_j: Dict[int, List[int]] = {}

    # also keep a mapping op_id -> op record
    op_by_id: Dict[int, dict] = {}

    for gk in group_keys:
        j = job_id_of_group[gk]
        ops_sorted = sorted(groups[gk], key=_op_sort_key)

        op_ids: List[int] = []
        for op in ops_sorted:
            oid = _safe_int(op.get("id"), None)
            if oid is None:
                continue
            if oid in op_by_id:
                # duplicate id in feed: make a new synthetic id
                # but you said ids are stable; still safe:
                new_id = max(I) + 1 if I else 1
                while new_id in op_by_id:
                    new_id += 1
                oid = new_id
            op_by_id[oid] = op
            op_ids.append(oid)
            I.append(oid)

        O_j[j] = op_ids

    # -------------------------
    # Build Pred_i: job-internal chain precedence
    # -------------------------
    Pred_i: Dict[int, List[int]] = {int(i): [] for i in I}
    for j, ops in O_j.items():
        for k in range(1, len(ops)):
            Pred_i[int(ops[k])] = [int(ops[k - 1])]

    # -------------------------
    # Feasible sets and flags
    # -------------------------
    M_i: Dict[int, List[int]] = {}
    L_i: Dict[int, List[int]] = {}
    beta_i: Dict[int, int] = {}
    p_i: Dict[int, float] = {}

    for i in I:
        op = op_by_id[i]

        # machines
        mc = op.get("machineCandidates", None)
        mc_list = _as_int_list(mc)
        if not mc_list:
            if allow_all_m:
                mc_list = list(M)
        else:
            # intersect with known machines if provided
            if M:
                mc_list = [m for m in mc_list if m in M] or (list(M) if allow_all_m else [])
        M_i[i] = mc_list

        # stations
        sc = op.get("stationCandidates", None)
        sc_list = _as_int_list(sc)
        if not sc_list:
            if allow_all_l:
                sc_list = list(L)
        else:
            if L:
                sc_list = [l for l in sc_list if l in L] or (list(L) if allow_all_l else [])
        L_i[i] = sc_list

        # station size requirement -> beta_i
        req = (op.get("stationSizeRequirement") or defaults.get("default_station_size_requirement") or "ANY")
        req_u = str(req).upper().strip()
        if req_u == "BIG":
            beta_i[i] = 1
        else:
            beta_i[i] = 0

        # processing time in hours
        ct = op.get("cycleTime", None)
        if ct is None:
            pt = default_pt
        else:
            try:
                pt = float(ct) / 60.0
            except Exception:
                pt = default_pt
        p_i[i] = max(0.0, float(pt))

    # -------------------------
    # r_j and d_j from plannedStart/End times (hours from plan_start)
    # -------------------------
    r_j: Dict[int, float] = {}
    d_j: Dict[int, float] = {}

    for j in J:
        ops = O_j.get(j, [])
        starts: List[float] = []
        ends: List[float] = []

        for i in ops:
            op = op_by_id[i]
            ps = _parse_iso_dt(op.get("plannedStartDateTime"))
            pe = _parse_iso_dt(op.get("plannedEndDateTime"))
            if ps:
                starts.append(_hours_between(plan_start_dt, ps))
            if pe:
                ends.append(_hours_between(plan_start_dt, pe))

        r_j[j] = float(min(starts)) if starts else 0.0
        # IMPORTANT: due dates should not be lost; if no plannedEnd exists use big number
        d_j[j] = float(max(ends)) if ends else 1e9

    # -------------------------
    # Post ops flags (keep from meta if exists, else default 0)
    # -------------------------
    # If you later want grinding/painting to come from operations, we can add.
    g_j = {int(j): int((base_meta.get("g_j") or {}).get(str(j), (base_meta.get("g_j") or {}).get(j, 0))) for j in J}
    p_flag_j = {int(j): int((base_meta.get("p_flag_j") or {}).get(str(j), (base_meta.get("p_flag_j") or {}).get(j, 0))) for j in J}
    t_grind_j = {int(j): float((base_meta.get("t_grind_j") or {}).get(str(j), (base_meta.get("t_grind_j") or {}).get(j, 0.0))) for j in J}
    t_paint_j = {int(j): float((base_meta.get("t_paint_j") or {}).get(str(j), (base_meta.get("t_paint_j") or {}).get(j, 0.0))) for j in J}

    # machine_type: optional
    machine_type = machines.get("machine_types", {}) or {}

    # Compose final solver input (main format)
    out: Dict[str, Any] = {
        "plan_calendar": plan_calendar,

        "J": J,
        "I": I,
        "M": M,
        "L": L,

        "L_big": L_big,
        "L_small": L_small,

        "machine_type": machine_type,

        "O_j": {int(k): [int(x) for x in v] for k, v in O_j.items()},
        "M_i": {int(k): [int(x) for x in v] for k, v in M_i.items()},
        "L_i": {int(k): [int(x) for x in v] for k, v in L_i.items()},
        "Pred_i": {int(k): [int(x) for x in v] for k, v in Pred_i.items()},

        "beta_i": {int(k): int(v) for k, v in beta_i.items()},

        # Use p_i; solver_core will expand to p_im automatically
        "p_i": {int(k): float(v) for k, v in p_i.items()},

        "r_j": {int(k): float(v) for k, v in r_j.items()},
        "d_j": {int(k): float(v) for k, v in d_j.items()},

        "g_j": {int(k): int(v) for k, v in g_j.items()},
        "p_flag_j": {int(k): int(v) for k, v in p_flag_j.items()},
        "t_grind_j": {int(k): float(v) for k, v in t_grind_j.items()},
        "t_paint_j": {int(k): float(v) for k, v in t_paint_j.items()},

        # keep original operations for debugging if you want
        "raw_operations": operations
    }

    # pass through k1 if provided
    if "k1" in base_meta:
        out["k1"] = base_meta["k1"]

    return out
