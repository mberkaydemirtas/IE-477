# -*- coding: utf-8 -*-
"""
adapter.py

Converts incoming 'operations' list (new single format) into solver-core data dict.

Key idea:
- Job grouping: parentId (preferred) or workOrderNumber fallback
- Precedence: derived from parentId group and erpOperationItemNumber order (chain)
- Stations: from stationCandidates + stationSizeRequirement
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import re


def _safe_int(x, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        return int(str(x).strip())
    except Exception:
        return default


def _parse_hhmmss(s: str) -> float:
    # returns hours
    if not s:
        return 0.0
    s = str(s).strip()
    if not s:
        return 0.0
    parts = s.split(":")
    if len(parts) != 3:
        return 0.0
    hh, mm, ss = parts
    try:
        return (int(hh) * 3600 + int(mm) * 60 + int(ss)) / 3600.0
    except Exception:
        return 0.0


def _cycle_minutes_to_hours(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        return float(v) / 60.0
    except Exception:
        return 0.0


def _norm_size_req(s: Any) -> str:
    s = (str(s).strip().upper() if s is not None else "ANY")
    if s in ["BIG", "L_BIG", "LARGE"]:
        return "BIG"
    if s in ["SMALL", "L_SMALL"]:
        return "SMALL"
    return "ANY"


def _extract_seq(op: dict) -> int:
    # precedence order inside a job
    # prefer erpOperationItemNumber, else try digits in name, else fallback to id
    x = op.get("erpOperationItemNumber", None)
    v = _safe_int(x, None)
    if v is not None:
        return v
    nm = str(op.get("name", "") or "")
    m = re.search(r"(\d+)", nm)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return int(_safe_int(op.get("id"), 0) or 0)


def build_data_from_operations(
    operations: List[dict],
    base_data: Dict[str, Any],
    plan_start_iso: str,
    plan_calendar: Optional[dict] = None,
    location_map: Optional[dict] = None,
) -> Dict[str, Any]:
    plan_calendar = plan_calendar or {"utc_offset": "+03:00"}
    location_map = location_map or {}

    # Defaults from base_data if candidates are missing
    default_M = [int(x) for x in (base_data.get("M") or [])]
    default_L = [int(x) for x in (base_data.get("L") or [])]

    if not default_M:
        default_M = [1, 2, 3, 4]
    if not default_L:
        default_L = [1, 2, 3, 4]

    L_big = [int(x) for x in (base_data.get("L_big") or [])]
    L_small = [int(x) for x in (base_data.get("L_small") or [])]

    # --- collect ops
    ops_clean: List[dict] = []
    for op in (operations or []):
        if not isinstance(op, dict):
            continue
        if op.get("isPlannable", True) is False:
            continue
        op_id = _safe_int(op.get("id"), None)
        if op_id is None:
            continue
        ops_clean.append(op)

    if not ops_clean:
        raise ValueError("No valid operations found in input")

    # --- group by job key (parentId preferred)
    # new format has parentId=job id (like JOB-1 etc). we convert to integer job indices later.
    groups: Dict[str, List[dict]] = {}
    for op in ops_clean:
        pid = op.get("parentId", None)
        job_key = None

        pid_int = _safe_int(pid, None)
        if pid_int is not None:
            job_key = f"PID:{pid_int}"
        else:
            wo = op.get("workOrderNumber", None)
            if wo:
                job_key = f"WO:{str(wo).strip()}"
            else:
                pn = op.get("parentName", None)
                job_key = f"PN:{str(pn).strip()}" if pn else "JOB:UNKNOWN"

        groups.setdefault(job_key, []).append(op)

    # map job_key -> job index 1..|J|
    job_keys = sorted(groups.keys())
    job_index_of: Dict[str, int] = {jk: idx + 1 for idx, jk in enumerate(job_keys)}

    # core sets
    J = list(range(1, len(job_keys) + 1))
    I = sorted([int(_safe_int(op["id"], 0) or 0) for op in ops_clean])

    # O_j and Pred_i
    O_j: Dict[int, List[int]] = {}
    Pred_i: Dict[int, List[int]] = {int(i): [] for i in I}

    # p_i (hours) and then solver_core will expand to p_im if needed
    p_i: Dict[int, float] = {}

    # M_i and L_i
    M_i: Dict[int, List[int]] = {}
    L_i: Dict[int, List[int]] = {}

    # beta_i: 1 if BIG required else 0
    beta_i: Dict[int, int] = {}

    # optional location / station mapping (if you want it later)
    # we keep it in data but heuristic ignores it unless you use it in adapter rules.
    op_location: Dict[int, Any] = {}

    for jk, ops in groups.items():
        j = job_index_of[jk]

        # order ops inside job by erpOperationItemNumber (or fallback)
        ops_sorted = sorted(ops, key=_extract_seq)
        op_ids = [int(_safe_int(o["id"], 0) or 0) for o in ops_sorted]
        O_j[j] = op_ids

        # chain precedence inside job: previous op -> next op
        for k in range(1, len(op_ids)):
            Pred_i[op_ids[k]].append(op_ids[k - 1])

        # fill per-op fields
        for o in ops_sorted:
            i = int(_safe_int(o["id"], 0) or 0)

            # processing time: cycleTime minutes -> hours
            p_i[i] = float(_cycle_minutes_to_hours(o.get("cycleTime", None)))

            # candidates
            mc = o.get("machineCandidates", None)
            sc = o.get("stationCandidates", None)

            if isinstance(mc, list) and mc:
                M_i[i] = [int(x) for x in mc]
            else:
                M_i[i] = list(default_M)

            if isinstance(sc, list) and sc:
                L_i[i] = [int(x) for x in sc]
            else:
                L_i[i] = list(default_L)

            # station size requirement
            req = _norm_size_req(o.get("stationSizeRequirement", "ANY"))
            beta_i[i] = 1 if req == "BIG" else 0

            # location info (optional)
            loc = None
            wc = o.get("workCenterName") or o.get("workCenterMachineCode")
            if wc and wc in location_map:
                loc = location_map[wc]
            elif wc:
                loc = wc
            op_location[i] = loc

    # r_j, d_j:
    # If plannedStartDateTime/plannedEndDateTime are present consistently in input, you can compute relative hours.
    # If not, keep 0 and VERY LARGE; your pipeline can override later.
    r_j: Dict[int, float] = {j: 0.0 for j in J}
    d_j: Dict[int, float] = {j: 1e9 for j in J}

    # post ops fields, default zeros unless base_data provides
    g_j = {j: int((base_data.get("g_j") or {}).get(str(j), (base_data.get("g_j") or {}).get(j, 0))) for j in J}
    p_flag_j = {j: int((base_data.get("p_flag_j") or {}).get(str(j), (base_data.get("p_flag_j") or {}).get(j, 0))) for j in J}
    t_grind_j = {j: float((base_data.get("t_grind_j") or {}).get(str(j), (base_data.get("t_grind_j") or {}).get(j, 0.0))) for j in J}
    t_paint_j = {j: float((base_data.get("t_paint_j") or {}).get(str(j), (base_data.get("t_paint_j") or {}).get(j, 0.0))) for j in J}

    # if base_data contains these sets, keep; else make from defaults
    M = list(default_M)
    L = list(default_L)

    out: Dict[str, Any] = {}

    # keep calendar fields if present
    out["plan_calendar"] = base_data.get("plan_calendar", {"utc_offset": "+03:00"})
    out["k1"] = float(base_data.get("k1", 2.0))

    # core keys
    out.update({
        "J": J,
        "I": I,
        "M": M,
        "L": L,
        "L_big": L_big if L_big else [L[0]],
        "L_small": L_small if L_small else [x for x in L if x not in (L_big if L_big else [])] or [],
        "O_j": O_j,
        "M_i": M_i,
        "L_i": L_i,
        "Pred_i": Pred_i,
        "p_i": p_i,            # NOTE: solver_core.normalize_data will expand p_i -> p_im
        "r_j": r_j,
        "d_j": d_j,
        "g_j": g_j,
        "p_flag_j": p_flag_j,
        "t_grind_j": t_grind_j,
        "t_paint_j": t_paint_j,
        "beta_i": beta_i,
        "op_location": op_location,  # optional metadata
        "job_key_map": job_index_of  # optional debug
    })

    return out
