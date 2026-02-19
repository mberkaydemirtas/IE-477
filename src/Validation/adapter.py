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


def _to_int_safe(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _to_float_safe(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _duration_hhmmss_to_hours(s: Optional[str]) -> float:
    # "HH:MM:SS" -> hours
    if not s:
        return 0.0
    try:
        parts = str(s).strip().split(":")
        if len(parts) != 3:
            return 0.0
        hh, mm, ss = parts
        return float(hh) + float(mm) / 60.0 + float(ss) / 3600.0
    except Exception:
        return 0.0


def _is_fason_op(op: dict) -> bool:
    machine_code = str(op.get("workCenterMachineCode") or "").strip().upper()
    machine_name = str(op.get("workCenterName") or "").strip().upper()
    return ("FASON" in machine_code) or ("FASON" in machine_name)


def build_data_from_operations(
    operations: List[dict],
    base_meta: Dict[str, Any],
    plan_start_iso: str,
    plan_calendar: Optional[dict] = None,
    location_map: Optional[dict] = None,
    system_config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Builds solver data (J,I,M,L,O_j,M_i,L_i,Pred_i,p_im,r_j,d_j,...) from operations list.

    AGREED RULES:
    - Job grouping: (erpCustomerOrderNumber, erpCustomerOrderItem) as one group.
      If erpCustomerOrderNumber is empty => fallback group key.
    - Within a workOrderNumber: operations follow erpOperationItemNumber order (Finish-to-Start).
    - Within the same customer group: different work orders scheduled by workOrderNumber ascending.
      We enforce it by linking last op of WO(k) -> first op of WO(k+1).
    - p_im (hours):
        setup + (plannedEndDateTime - plannedStartDateTime)   if both planned timestamps exist
        else setup + (cycleTime * plannedPartCount)           if cycleTime is not null and >0
      NOTE: cycleTime assumed MINUTES / part.
    - plannedEndDateTime is NOT used for due dates.
    - d_j comes from endDate (group deadline). If multiple, use MIN endDate.
    - isPlannable == False => anchored: fixed_ops uses plannedStartDateTime/plannedEndDateTime times (relative hours).
    - IMPORTANT FIX (you requested): M/L universe should match the data if system_config is missing
      OR does not cover ids in data. We derive M/L from workCenterMachineId/workCenterId.
    """

    # merge system_config -> base_meta (system_config first, then base_meta overrides)
    if isinstance(system_config, dict):
        base_meta = _deep_merge(system_config, base_meta)

    plan_calendar = plan_calendar or base_meta.get("plan_calendar") or base_meta.get("calendar") or {"utc_offset": "+03:00"}
    location_map = location_map or {}

    plan_start_dt = _parse_iso(plan_start_iso) or datetime.now(timezone.utc)

    # -------------------------
    # Filter only OPERATION dicts
    # -------------------------
    ops_in: List[dict] = []
    for op in operations or []:
        if isinstance(op, dict) and (op.get("objectType") in (None, "OPERATION")):
            ops_in.append(op)

    # -------------------------
    # Build M / L universe
    # -------------------------
    machine_ids_cfg = _deep_get(base_meta, ["machines", "machine_ids"], None) or base_meta.get("M")
    station_ids_cfg = _deep_get(base_meta, ["stations", "station_ids"], None) or base_meta.get("L")

    data_machine_ids = sorted({
        _to_int_safe(o.get("workCenterMachineId"), 0)
        for o in ops_in
        if _to_int_safe(o.get("workCenterMachineId"), 0) > 0
    })
    data_station_ids = sorted({
        _to_int_safe(o.get("workCenterId"), 0)
        for o in ops_in
        if _to_int_safe(o.get("workCenterId"), 0) > 0
    })

    # if config missing => use data ids (or fallback)
    if not machine_ids_cfg:
        machine_ids_cfg = data_machine_ids or [1, 2, 3, 4]
    if not station_ids_cfg:
        station_ids_cfg = data_station_ids or [1, 2, 3, 4]

    # if config exists but does NOT cover data ids => expand with data ids
    try:
        cfgM = sorted({int(x) for x in (machine_ids_cfg or [])})
    except Exception:
        cfgM = []
    try:
        cfgL = sorted({int(x) for x in (station_ids_cfg or [])})
    except Exception:
        cfgL = []

    if data_machine_ids:
        missingM = [m for m in data_machine_ids if m not in cfgM]
        if missingM:
            cfgM = sorted(set(cfgM) | set(data_machine_ids))
    if data_station_ids:
        missingL = [l for l in data_station_ids if l not in cfgL]
        if missingL:
            cfgL = sorted(set(cfgL) | set(data_station_ids))

    M = cfgM if cfgM else (data_machine_ids or [1, 2, 3, 4])
    L = cfgL if cfgL else (data_station_ids or [1, 2, 3, 4])

    M = [int(x) for x in M]
    L = [int(x) for x in L]

    M_set = set(int(x) for x in M)
    L_set = set(int(x) for x in L)

    # Each fason operation gets its own dedicated machine and station ids.
    fason_machine_of_raw_op: Dict[int, int] = {}
    fason_station_of_raw_op: Dict[int, int] = {}
    next_fason_machine = (max(M) + 1) if M else 1
    next_fason_station = (max(L) + 1) if L else 1
    for op in ops_in:
        if not _is_fason_op(op):
            continue
        raw_op_id = _to_int_safe(op.get("id"), 0)
        if raw_op_id <= 0 or raw_op_id in fason_machine_of_raw_op:
            continue

        fason_machine_of_raw_op[raw_op_id] = next_fason_machine
        M.append(next_fason_machine)
        M_set.add(next_fason_machine)
        next_fason_machine += 1

        fason_station_of_raw_op[raw_op_id] = next_fason_station
        L.append(next_fason_station)
        L_set.add(next_fason_station)
        next_fason_station += 1

    # station type -> big/small (kept for compatibility)
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
    min_pt_hours = float(defaults.get("min_processing_time_hours", 1.0 / 60.0))  # 1 minute default
    use_planned_release_times = bool(base_meta.get("use_planned_release_times", False))
    internal_machines = [m for m in range(1, 13) if m in M_set]

    # -------------------------
    # Grouping (Job definition)
    # -------------------------
    def _customer_key(op: dict) -> Tuple[str, str]:
        co = str(op.get("erpCustomerOrderNumber") or "").strip()
        item = str(op.get("erpCustomerOrderItem") or "").strip()

        # fallback: if customer order missing, group by (NO_CO, partNumber)
        if co == "":
            pn = str(op.get("partNumber") or "NO_PART").strip()
            return ("NO_CO", pn)
        if item == "":
            item = "0"
        return (co, item)

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for op in ops_in:
        groups.setdefault(_customer_key(op), []).append(op)

    group_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    job_of_group = {gk: idx + 1 for idx, gk in enumerate(group_keys)}
    J = list(range(1, len(group_keys) + 1))

    # outputs
    I: List[int] = []
    O_j: Dict[int, List[int]] = {j: [] for j in J}

    M_i: Dict[int, List[int]] = {}
    L_i: Dict[int, List[int]] = {}
    beta_i: Dict[int, int] = {}
    Pred_i: Dict[int, List[int]] = {}

    p_im: Dict[Tuple[int, int], float] = {}

    r_j: Dict[int, float] = {}
    d_j: Dict[int, float] = {}
    g_j: Dict[int, int] = {}
    p_flag_j: Dict[int, int] = {}
    t_grind_j: Dict[int, float] = {}
    t_paint_j: Dict[int, float] = {}

    # anchored ops
    fixed_ops: Dict[int, Dict[str, Any]] = {}

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

    fallback_next = 1

    def _gen_fallback_id() -> int:
        nonlocal fallback_next
        while fallback_next in used_ids:
            fallback_next += 1
        used_ids.add(fallback_next)
        return fallback_next

    def _hours_from_plan(dt: datetime) -> float:
        return max(0.0, (dt - plan_start_dt).total_seconds() / 3600.0)

    def _compute_pt_hours(op: dict) -> float:
        setup_h = _duration_hhmmss_to_hours(op.get("plannedSetupDuration"))
        qty = _to_int_safe(op.get("plannedPartCount"), default=0)

        st = _parse_iso(op.get("plannedStartDateTime"))
        en = _parse_iso(op.get("plannedEndDateTime"))
        if st and en:
            win_s = (en - st).total_seconds()
            win_h = max(0.0, win_s / 3600.0)
            return max(setup_h + win_h, min_pt_hours)

        ct = op.get("cycleTime", None)
        ct_val = 0.0
        if ct is not None:
            sct = str(ct).strip()
            if sct != "":
                ct_val = _to_float_safe(ct, default=0.0)
            if ct_val > 0.0 and qty > 0:
                # cycleTime assumed minutes/part
                return max(setup_h + (ct_val * qty) / 60.0, min_pt_hours)

        return max(setup_h + default_pt_hours, min_pt_hours)

    def _machine_candidates(op: dict, op_id: int) -> List[int]:
        # Fason operations must never consume internal machine capacity.
        if _is_fason_op(op):
            fm = fason_machine_of_raw_op.get(int(op_id))
            if fm is not None:
                return [int(fm)]

        # Non-fason operations are restricted to internal machines 1..12.
        mid = op.get("workCenterMachineId", None)
        if mid is not None:
            m = _to_int_safe(mid, default=0)
            if m > 0 and m in internal_machines:
                return [m]

        # fallback: explicit candidates
        mc = op.get("machineCandidates", None) or op.get("feasible_machines", None)
        if mc:
            try:
                out = sorted({int(x) for x in mc if _to_int_safe(x, 0) in internal_machines})
                if out:
                    return out
            except Exception:
                pass

        return list(internal_machines) if allow_all_m else []

    def _station_candidates(op: dict) -> List[int]:
        if _is_fason_op(op):
            raw_op_id = _to_int_safe(op.get("id"), 0)
            fs = fason_station_of_raw_op.get(int(raw_op_id))
            if fs is not None:
                return [int(fs)]

        sid = op.get("workCenterId", None)
        if sid is not None:
            l = _to_int_safe(sid, default=0)
            if l > 0 and l in L_set:
                return [l]

        sc = op.get("stationCandidates", None) or op.get("feasible_stations", None)
        if sc:
            try:
                out = sorted({int(x) for x in sc if _to_int_safe(x, 0) in L_set})
                if out:
                    return out
            except Exception:
                pass

        return list(L) if allow_all_l else []

    # -------------------------
    # Build each job/group
    # -------------------------
    for gk in group_keys:
        j = job_of_group[gk]
        ops = groups[gk]

        by_wo: Dict[str, List[dict]] = {}
        for op in ops:
            wo = str(op.get("workOrderNumber") or "").strip()
            if wo == "":
                wo = "NO_WO"
            by_wo.setdefault(wo, []).append(op)

        def _wo_sort_key(wo: str):
            return (_to_int_safe(wo, default=10**18), wo)

        wo_keys = sorted(by_wo.keys(), key=_wo_sort_key)

        job_op_ids: List[int] = []
        group_start_times: List[datetime] = []
        group_due_times: List[datetime] = []

        last_op_of_prev_wo: Optional[Tuple[int, int, bool]] = None

        for wo in wo_keys:
            wo_ops = by_wo[wo]

            def _op_item_sort_key(o: dict):
                return (
                    _to_int_safe(o.get("erpOperationItemNumber"), default=10**9),
                    _to_int_safe(o.get("id"), default=10**9),
                )

            wo_ops_sorted = sorted(wo_ops, key=_op_item_sort_key)
            wo_op_ids: List[int] = []
            wo_op_meta: List[Tuple[int, int, bool]] = []  # (op_id, seq, is_fason)

            for op in wo_ops_sorted:
                oid = _reserve_id(op.get("id"))
                if oid is None:
                    oid = _gen_fallback_id()

                I.append(oid)
                job_op_ids.append(oid)
                wo_op_ids.append(oid)
                seq_no = _to_int_safe(op.get("erpOperationItemNumber"), default=10**9)
                is_fason = _is_fason_op(op)
                wo_op_meta.append((oid, seq_no, is_fason))

                M_i[oid] = _machine_candidates(op, oid)
                L_i[oid] = _station_candidates(op)

                ssr = str(op.get("stationSizeRequirement", "ANY")).upper()
                beta_i[oid] = 1 if ssr == "BIG" else 0

                pt_h = _compute_pt_hours(op)
                for m in M_i[oid]:
                    p_im[(oid, int(m))] = float(pt_h)

                Pred_i[oid] = []

                st = _parse_iso(op.get("plannedStartDateTime"))
                if st:
                    group_start_times.append(st)

                due_dt = _parse_iso(op.get("endDate"))
                if due_dt:
                    group_due_times.append(due_dt)

                is_plannable = op.get("isPlannable", True)
                if is_plannable is False:
                    st2 = _parse_iso(op.get("plannedStartDateTime"))
                    en2 = _parse_iso(op.get("plannedEndDateTime"))
                    if st2 and en2:
                        row: Dict[str, Any] = {
                            "start": _hours_from_plan(st2),
                            "finish": _hours_from_plan(en2),
                        }
                        if M_i[oid]:
                            row["machine"] = int(M_i[oid][0])
                        if L_i[oid]:
                            row["station"] = int(L_i[oid][0])
                        fixed_ops[oid] = row

            # Default chain only for increasing sequence numbers.
            # If both adjacent ops are fason, don't force precedence.
            for k in range(1, len(wo_op_meta)):
                prev_id, prev_seq, prev_fason = wo_op_meta[k - 1]
                cur_id, cur_seq, cur_fason = wo_op_meta[k]
                if cur_seq <= prev_seq:
                    continue
                if prev_fason and cur_fason:
                    continue
                Pred_i[cur_id].append(prev_id)

            if last_op_of_prev_wo is not None and wo_op_meta:
                cur_first_id, _, cur_first_fason = wo_op_meta[0]
                prev_last_id, _, prev_last_fason = last_op_of_prev_wo
                if not (prev_last_fason and cur_first_fason):
                    Pred_i[cur_first_id].append(prev_last_id)

            if wo_op_meta:
                last_op_of_prev_wo = wo_op_meta[-1]

        O_j[j] = job_op_ids

        if use_planned_release_times and group_start_times:
            rj = _hours_from_plan(min(group_start_times))
        else:
            rj = 0.0
        r_j[j] = float(rj)

        if group_due_times:
            dj = _hours_from_plan(min(group_due_times))
        else:
            if job_op_ids:
                dj = rj + sum(
                    p_im[(opx, M_i[opx][0])]
                    for opx in job_op_ids
                    if M_i.get(opx)
                )
            else:
                dj = rj
        d_j[j] = float(dj)

        g_j[j] = 0
        p_flag_j[j] = 0
        t_grind_j[j] = 0.0
        t_paint_j[j] = 0.0

    # machine types (keep compatibility)
    machine_type = base_meta.get("machine_type", None)
    if not isinstance(machine_type, dict):
        mt = _deep_get(base_meta, ["machines", "machine_types"], {}) or {}
        machine_type = {}
        for m in M:
            t = str(mt.get(str(m), mt.get(m, "TIG"))).upper()
            machine_type[str(m)] = 1 if t == "TIG" else 2

    out = dict(base_meta)
    machine_label_map = {str(int(m)): str(int(m)) for m in M}
    station_label_map = {str(int(l)): str(int(l)) for l in L}

    for k, machine_id in enumerate(sorted(fason_machine_of_raw_op.values()), start=1):
        machine_label_map[str(int(machine_id))] = f"Fason {k}"

    for k, station_id in enumerate(sorted(fason_station_of_raw_op.values()), start=1):
        station_label_map[str(int(station_id))] = f"Fason {k}"

    out.update({
        "J": J,
        "I": sorted([int(x) for x in I]),
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

        "p_im": {f"{i},{m}": p for (i, m), p in p_im.items()},

        "r_j": r_j,
        "d_j": d_j,

        "g_j": g_j,
        "p_flag_j": p_flag_j,
        "t_grind_j": t_grind_j,
        "t_paint_j": t_paint_j,

        "fixed_ops": fixed_ops,  # anchored ops
        "machine_label_map": machine_label_map,
        "station_label_map": station_label_map,
        "default_processing_time_hours": default_pt_hours,
        "plan_calendar": plan_calendar if isinstance(plan_calendar, dict) else {"utc_offset": "+03:00"},
    })

    return out
