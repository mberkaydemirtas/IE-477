#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Set


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    s2 = s.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _deep_get(d: dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_data_from_operations(
    operations: List[dict],
    base_meta: Dict[str, Any],
    plan_start_iso: str,
    plan_calendar: Optional[dict] = None,
    location_map: Optional[dict] = None,
    system_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Builds OLD solver data (J,I,M,L,O_j,M_i,L_i,Pred_i,p_im,r_j,d_j,...) from
    NEW operations list.

    IMPORTANT FIXES:
    - Processing time is NOT inferred from plannedStart/End differences.
      Only cycleTime (minutes) or default is used.
    - Operation IDs are stable across baseline/reschedule:
      uses op["id"] if possible, otherwise a generated unique ID.
    - Due dates (d_j) are still derived from plannedEndDateTime (job max end),
      relative to plan_start_iso (as hours).
    """

    # merge system_config -> base_meta (system_config first, then base_meta overrides)
    if isinstance(system_config, dict):
        base_meta = _deep_merge(system_config, base_meta)

    plan_calendar = plan_calendar or base_meta.get("plan_calendar") or base_meta.get("calendar") or {"utc_offset": "+03:00"}
    location_map = location_map or {}

    plan_start_dt = _parse_iso(plan_start_iso) or datetime.now(timezone.utc)

    # --- system config / base meta ---
    machine_ids = _deep_get(base_meta, ["machines", "machine_ids"], None) or base_meta.get("M")
    station_ids = _deep_get(base_meta, ["stations", "station_ids"], None) or base_meta.get("L")

    if not machine_ids:
        machine_ids = [1, 2, 3, 4]
    if not station_ids:
        station_ids = [1, 2, 3, 4]

    M = [int(x) for x in machine_ids]
    L = [int(x) for x in station_ids]

    # station type -> big/small
    station_types = _deep_get(base_meta, ["stations", "station_types"], {}) or {}
    L_big: List[int] = []
    L_small: List[int] = []
    for l in L:
        t = str(station_types.get(str(l), station_types.get(l, "ANY"))).upper()
        if t == "BIG":
            L_big.append(l)
        else:
            L_small.append(l)

    if not L_big:
        L_big = L[:3] if len(L) >= 3 else L[:]
        L_small = [x for x in L if x not in L_big]

    defaults = base_meta.get("defaults", {}) if isinstance(base_meta.get("defaults"), dict) else {}
    allow_all_m = bool(defaults.get("allow_all_machines_if_missing", True))
    allow_all_l = bool(defaults.get("allow_all_stations_if_missing", True))
    default_pt_hours = float(defaults.get("default_processing_time_hours", 1.0))

    # --- group ops by parentId -> job ---
    groups: Dict[int, List[dict]] = {}
    for op in operations:
        pid = op.get("parentId", None)
        if pid is None:
            pid = op.get("workOrderNumber", None)
        try:
            pid_int = int(pid) if pid is not None else -1
        except Exception:
            pid_int = -1
        groups.setdefault(pid_int, []).append(op)

    parent_keys = sorted(groups.keys())
    job_of_parent = {pk: idx + 1 for idx, pk in enumerate(parent_keys)}
    J = list(range(1, len(parent_keys) + 1))

    # outputs
    I: List[int] = []
    O_j: Dict[int, List[int]] = {j: [] for j in J}

    M_i: Dict[int, List[int]] = {}
    L_i: Dict[int, List[int]] = {}
    beta_i: Dict[int, int] = {}
    p_im: Dict[Tuple[int, int], float] = {}
    Pred_i: Dict[int, List[int]] = {}

    r_j: Dict[int, float] = {}
    d_j: Dict[int, float] = {}
    g_j: Dict[int, int] = {}
    p_flag_j: Dict[int, int] = {}
    t_grind_j: Dict[int, float] = {}
    t_paint_j: Dict[int, float] = {}

    # stable op ids
    used_ids: Set[int] = set()

    def _reserve_id(candidate: Optional[Any]) -> Optional[int]:
        if candidate is None:
            return None
        try:
            cid = int(candidate)
        except Exception:
            return None
        if cid <= 0:
            return None
        if cid in used_ids:
            return None
        used_ids.add(cid)
        return cid

    # fallback id generator
    fallback_next = 1

    def _gen_fallback_id() -> int:
        nonlocal fallback_next
        while fallback_next in used_ids:
            fallback_next += 1
        used_ids.add(fallback_next)
        return fallback_next

    for pk in parent_keys:
        j = job_of_parent[pk]
        ops = groups[pk]

        # keep old plan sorting (optional)
        def _sort_key(o):
            dt = _parse_iso(o.get("plannedStartDateTime"))
            return dt or datetime(1970, 1, 1, tzinfo=timezone.utc)

        ops_sorted = sorted(ops, key=_sort_key)

        job_op_ids: List[int] = []
        start_times: List[datetime] = []
        end_times: List[datetime] = []

        for op in ops_sorted:
            # ✅ stable operation id
            oid = _reserve_id(op.get("id"))
            if oid is None:
                oid = _gen_fallback_id()

            I.append(oid)
            job_op_ids.append(oid)

            # feasible machines
            mc = op.get("machineCandidates", None)
            if mc is None:
                mc = op.get("feasible_machines", None)
            if mc is None and allow_all_m:
                mc = M
            mc_list = [int(x) for x in (mc or [])]
            if not mc_list:
                mc_list = list(M)
            M_i[oid] = mc_list

            # feasible stations
            sc = op.get("stationCandidates", None)
            if sc is None:
                sc = op.get("feasible_stations", None)
            if sc is None and allow_all_l:
                sc = L
            sc_list = [int(x) for x in (sc or [])]
            if not sc_list:
                sc_list = list(L)
            L_i[oid] = sc_list

            # BIG requirement
            ssr = str(op.get("stationSizeRequirement", "ANY")).upper()
            beta_i[oid] = 1 if ssr == "BIG" else 0

            # --- processing time (hours): ONLY cycleTime or default ---
            pt_hours = None
            if op.get("cycleTime", None) is not None:
                try:
                    pt_hours = float(op["cycleTime"]) / 60.0
                except Exception:
                    pt_hours = None

            if pt_hours is None or pt_hours <= 0:
                pt_hours = default_pt_hours

            for m in M_i[oid]:
                p_im[(oid, int(m))] = float(pt_hours)

            # collect planned times for r/d only
            st = _parse_iso(op.get("plannedStartDateTime"))
            en = _parse_iso(op.get("plannedEndDateTime"))
            if st:
                start_times.append(st)
            if en:
                end_times.append(en)

            Pred_i[oid] = []

        # chain precedence within job (temporary rule)
        for k in range(1, len(job_op_ids)):
            Pred_i[job_op_ids[k]].append(job_op_ids[k - 1])

        O_j[j] = job_op_ids

        # release/due times from planned windows (NOT duration)
        if start_times:
            rj = max(0.0, (min(start_times) - plan_start_dt).total_seconds() / 3600.0)
        else:
            rj = 0.0

        if end_times:
            dj = max(0.0, (max(end_times) - plan_start_dt).total_seconds() / 3600.0)
        else:
            # fallback: release + sum of default durations
            dj = rj + sum(p_im[(opx, M_i[opx][0])] for opx in job_op_ids)

        r_j[j] = float(rj)
        d_j[j] = float(dj)

        # post-ops defaults (you can derive later from op names if needed)
        g_j[j] = 0
        p_flag_j[j] = 0
        t_grind_j[j] = 0.0
        t_paint_j[j] = 0.0

    # machine types
    machine_type = base_meta.get("machine_type", None)
    if not isinstance(machine_type, dict):
        mt = _deep_get(base_meta, ["machines", "machine_types"], {}) or {}
        machine_type = {}
        for m in M:
            t = str(mt.get(str(m), mt.get(m, "TIG"))).upper()
            machine_type[str(m)] = 1 if t == "TIG" else 2

    out = dict(base_meta)
    out.update({
        "J": J,
        "I": sorted(I),
        "M": M,
        "L": L,
        "L_big": L_big,
        "L_small": L_small,
        "machine_type": machine_type,

        "O_j": O_j,
        "M_i": M_i,
        "L_i": L_i,
        "Pred_i": Pred_i,
        "beta_i": beta_i,

        # keep JSON-friendly p_im keys
        "p_im": {f"{i},{m}": p for (i, m), p in p_im.items()},

        "r_j": r_j,
        "d_j": d_j,

        "g_j": g_j,
        "p_flag_j": p_flag_j,
        "t_grind_j": t_grind_j,
        "t_paint_j": t_paint_j,

        "default_processing_time_hours": default_pt_hours,
        "plan_calendar": plan_calendar if isinstance(plan_calendar, dict) else {"utc_offset": "+03:00"},
    })

    return out
