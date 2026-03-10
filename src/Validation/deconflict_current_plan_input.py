#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _extract_operations(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("operations", "assignments", "items", "data", "workOrderOperationDtoList"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


def _parse_dt(value):
    if isinstance(value, (list, tuple)):
        parts = list(value)
        if len(parts) >= 3:
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3]) if len(parts) >= 4 else 0
            minute = int(parts[4]) if len(parts) >= 5 else 0
            second = int(parts[5]) if len(parts) >= 6 else 0
            microsecond = 0
            if len(parts) >= 7:
                microsecond = max(0, min(999999, int(parts[6]) // 1000))
            return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _to_list_dt(dt: datetime):
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond * 1000]


def main():
    if len(sys.argv) < 3:
        print("Usage: py deconflict_current_plan_input.py input.json output.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    raw = _load_json(input_path)
    out = copy.deepcopy(raw)
    ops = _extract_operations(out)
    if not ops:
        raise ValueError("No operations list found in input")

    grouped = defaultdict(list)
    for idx, op in enumerate(ops):
        machine_id = op.get("workCenterMachineId")
        grouped[machine_id].append((idx, op))

    shifted = 0
    for machine_id, rows in grouped.items():
        enriched = []
        for idx, op in rows:
            st = _parse_dt(op.get("plannedStartDateTime"))
            en = _parse_dt(op.get("plannedEndDateTime"))
            if st is None or en is None:
                continue
            if en < st:
                en = st
            enriched.append((idx, op, st, en, en - st))

        enriched.sort(key=lambda x: (x[2], x[3], int(x[1].get("id", 0))))
        prev_end = None
        for idx, op, st, en, dur in enriched:
            new_start = st if prev_end is None or st >= prev_end else prev_end
            new_end = new_start + dur
            if new_start != st or new_end != en:
                shifted += 1
                op["plannedStartDateTime"] = _to_list_dt(new_start)
                op["plannedEndDateTime"] = _to_list_dt(new_end)
            prev_end = new_end

    _save_json(output_path, out)
    print(f"Saved: {output_path}")
    print(f"Shifted operations: {shifted}")


if __name__ == "__main__":
    main()
