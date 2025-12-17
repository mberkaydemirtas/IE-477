#!/usr/bin/env python3
# -- coding: utf-8 --

import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

from solver_core import (
    make_base_data,
    solve_baseline,
    solve_reschedule,
)

app = FastAPI(title="Welding Scheduling API", version="1.0")

BASELINE_PATH = Path("baseline_solution.json")


# ----------------------------
# Request Schemas
# ----------------------------

class BaselineRequest(BaseModel):
    # ileride gerçek veri göndereceksiniz; şimdilik demo data kullanıyoruz
    save_to_disk: bool = True


class RescheduleUrgentConfig(BaseModel):
    job_id: int = 9
    ops_count: int = 3
    due_slack: float = 10.0


class RescheduleRequest(BaseModel):
    # UI'dan gelecek alanlar
    mode: Literal["continue", "optimize"] = "continue"
    shift_start_hhmm: str = "08:00"
    unavailable_machines: List[int] = Field(default_factory=list)
    unavailable_stations: List[int] = Field(default_factory=list)
    urgent: RescheduleUrgentConfig = Field(default_factory=RescheduleUrgentConfig)

    # baseline nereden okunacak
    baseline_source: Literal["disk"] = "disk"


# ----------------------------
# Endpoints
# ----------------------------

@app.post("/schedule/baseline")
def schedule_baseline(req: BaselineRequest):
    data = make_base_data()
    try:
        sol = solve_baseline(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Baseline solve failed: {e}")

    if req.save_to_disk:
        BASELINE_PATH.write_text(json.dumps(sol, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "saved": bool(req.save_to_disk),
        "objective": sol["objective"],
        "baseline_file": str(BASELINE_PATH) if req.save_to_disk else None
    }


@app.post("/schedule/reschedule")
def schedule_reschedule(req: RescheduleRequest):
    data = make_base_data()

    # load baseline solution
    if req.baseline_source == "disk":
        if not BASELINE_PATH.exists():
            raise HTTPException(
                status_code=400,
                detail="No baseline_solution.json found. Run /schedule/baseline first."
            )
        old_solution = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    else:
        raise HTTPException(status_code=400, detail="Unsupported baseline_source")

    # reschedule
    try:
        res = solve_reschedule(
            data_base=data,
            old_solution=old_solution,
            shift_start_hhmm=req.shift_start_hhmm,
            unavailable_machines=req.unavailable_machines,
            unavailable_stations=req.unavailable_stations,
            mode=req.mode,
            urgent_job_id=req.urgent.job_id,
            urgent_ops_count=req.urgent.ops_count,
            urgent_due_slack=req.urgent.due_slack
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reschedule failed: {e}")

    # (opsiyonel) reschedule sonucunu da kaydedebilirsin
    Path("reschedule_solution.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "t0": res["t0"],
        "sets": res["sets"],
        "objective": res["objective"],
        "urgent": res["urgent"],
        "keep_decisions": res["keep_decisions"],
        "saved_reschedule_file": "reschedule_solution.json"
    }
