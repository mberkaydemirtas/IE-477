#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Welding Shop Scheduling – LARGE EXAMPLE (8 jobs, 30 ops)

RESCHEDULING EXTENSION
STEP 1: compute t0 automatically + store old solution
STEP 2: classify done/run/free + freeze framework + availability inputs
STEP 3: add urgent job at t0 and solve rescheduling (objective unchanged: min T_max)
STEP 4: add "model decides" option for RUNNING ops (keep binary)
STEP 5: TRUE PREEMPTION (SPLIT) for RUNNING ops when keep=0:
        - freeze done-part to [S_old, t0]
        - create a remainder operation with remaining processing time, scheduled after t0
        - precedence & job completion updated robustly (C_weld as max over job ops)
"""

from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ==========================================================
#  RESCHEDULING – HELPERS
# ==========================================================

def compute_t0_hours(shift_start_hhmm="08:00"):
    """t0 = (now - shift_start) in HOURS"""
    now = datetime.now()
    hh, mm = map(int, shift_start_hhmm.split(":"))
    shift_start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if now < shift_start:
        shift_start = shift_start - timedelta(days=1)
    return float((now - shift_start).total_seconds() / 3600.0)


def extract_old_solution(I, x, y, S, C):
    """Store old schedule values from baseline solve."""
    S_old = {i: float(S[i].X) for i in I}
    C_old = {i: float(C[i].X) for i in I}
    x_old = {(i, m): int(round(x[i, m].X)) for (i, m) in x.keys()}
    y_old = {(i, l): int(round(y[i, l].X)) for (i, l) in y.keys()}
    return S_old, C_old, x_old, y_old


def classify_ops_by_t0(I, S_old, C_old, t0):
    """
    done: C_old <= t0
    run : S_old < t0 < C_old
    free: S_old >= t0
    """
    done = [i for i in I if C_old[i] <= t0 + 1e-9]
    run  = [i for i in I if (S_old[i] < t0 - 1e-9) and (C_old[i] > t0 + 1e-9)]
    free = [i for i in I if S_old[i] >= t0 - 1e-9]
    return done, run, free


def parse_int_list(user_str):
    s = (user_str or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def add_urgent_job(data, t0, urgent_job_id=9, urgent_ops_count=3, urgent_due_slack=10.0):
    """
    Add urgent job u at time t0.
    - r_u = t0
    - d_u = t0 + urgent_due_slack
    - precedence: chain + last depends on all
    - feasibility: same odd/even TIG/MAG rule
    - beta: last urgent op requires big station
    """
    J, I, O_j = data["J"], data["I"], data["O_j"]
    M, L = data["M"], data["L"]
    machine_type, K_i, M_i = data["machine_type"], data["K_i"], data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    r_j, d_j = data["r_j"], data["d_j"]
    g_j, p_j = data["g_j"], data["p_j"]
    t_grind_j, t_paint_j = data["t_grind_j"], data["t_paint_j"]
    beta_i = data["beta_i"]

    if urgent_job_id in J:
        raise ValueError("urgent_job_id already exists in J.")

    next_op = max(I) + 1
    new_ops = list(range(next_op, next_op + urgent_ops_count))

    J2 = J[:] + [urgent_job_id]
    I2 = I[:] + new_ops
    O_j2 = dict(O_j)
    O_j2[urgent_job_id] = new_ops

    # stations feasible
    L_i2 = dict(L_i)
    for i in new_ops:
        L_i2[i] = L[:]

    # machine feasibility + processing times
    K_i2 = dict(K_i)
    M_i2 = dict(M_i)
    p_im2 = dict(p_im)

    for i in new_ops:
        K_i2[i] = [1] if (i % 2 == 1) else [2]
        M_i2[i] = [m for m in M if machine_type[m] in K_i2[i]]
        for m in M_i2[i]:
            base = 3 + (i % 5)
            machine_add = (m - 1) * 0.5
            p_im2[(i, m)] = float(base + machine_add)

    # precedence
    Pred_i2 = {i: list(Pred_i[i]) for i in Pred_i}
    for i in new_ops:
        Pred_i2[i] = []

    for k in range(1, len(new_ops)):
        Pred_i2[new_ops[k]] = [new_ops[k - 1]]

    last = new_ops[-1]
    preds = set(Pred_i2[last])
    for op in new_ops:
        if op != last:
            preds.add(op)
    Pred_i2[last] = list(preds)

    # release & due
    r_j2 = dict(r_j)
    d_j2 = dict(d_j)
    r_j2[urgent_job_id] = float(t0)
    d_j2[urgent_job_id] = float(t0 + urgent_due_slack)

    # grind/paint off
    g_j2 = dict(g_j)
    p_j2 = dict(p_j)
    t_grind_j2 = dict(t_grind_j)
    t_paint_j2 = dict(t_paint_j)

    g_j2[urgent_job_id] = 0
    p_j2[urgent_job_id] = 0
    t_grind_j2[urgent_job_id] = 0.0
    t_paint_j2[urgent_job_id] = 0.0

    # beta
    beta_i2 = dict(beta_i)
    for i in new_ops:
        beta_i2[i] = 0
    beta_i2[last] = 1

    data2 = dict(data)
    data2.update({
        "J": J2,
        "I": I2,
        "O_j": O_j2,
        "K_i": K_i2,
        "M_i": M_i2,
        "L_i": L_i2,
        "Pred_i": Pred_i2,
        "p_im": p_im2,
        "r_j": r_j2,
        "d_j": d_j2,
        "g_j": g_j2,
        "p_j": p_j2,
        "t_grind_j": t_grind_j2,
        "t_paint_j": t_paint_j2,
        "beta_i": beta_i2,
        "urgent_job_id": urgent_job_id,
        "urgent_ops": new_ops
    })
    return data2


def add_split_remainders_for_running_ops(data, I_run, t0, old_solution):
    """
    STEP 5:
    For each running op i:
      - create new op i_rem (remainder)
      - remainder processing times = remaining_time = p(i, m_old) - (t0 - S_old)
      - add precedence: i_rem depends on i  (so it starts after done-part ends)
      - redirect successors: if k has pred i, also require k after i_rem (soft via adding pred)
        (This makes the split effective; keep=1 case will freeze i and i_rem can be ignored.)
    """
    J, I, O_j = data["J"], data["I"], data["O_j"]
    M, L = data["M"], data["L"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    beta_i = data["beta_i"]

    S_old = old_solution["S_old"]
    x_old = old_solution["x_old"]

    next_op = max(I) + 1
    rem_map = {}  # i -> i_rem

    # find job of each op (for O_j update)
    op_to_job = {}
    for j in J:
        for op in O_j[j]:
            op_to_job[op] = j

    I2 = I[:]
    O_j2 = {j: list(O_j[j]) for j in J}
    L_i2 = dict(L_i)
    M_i2 = dict(M_i)
    Pred_i2 = {k: list(Pred_i[k]) for k in Pred_i}
    p_im2 = dict(p_im)
    beta_i2 = dict(beta_i)

    # helper: old selected machine for i
    def old_machine(i):
        for (ii, m), v in x_old.items():
            if ii == i and v == 1:
                return m
        return None

    for i in I_run:
        i_rem = next_op
        next_op += 1
        rem_map[i] = i_rem

        # add to global ops
        I2.append(i_rem)

        # put remainder into same job list (append at end is fine because C_weld uses max now)
        j = op_to_job.get(i, None)
        if j is not None:
            O_j2[j].append(i_rem)

        # same feasible stations/machines as original
        L_i2[i_rem] = list(L_i[i])
        M_i2[i_rem] = list(M_i[i])

        # remaining time based on old machine
        m0 = old_machine(i)
        if m0 is None:
            # fallback: take first feasible machine
            m0 = M_i[i][0]
        full_p = p_im[(i, m0)]
        elapsed = max(0.0, t0 - S_old[i])
        remaining = max(0.01, full_p - elapsed)  # minimum epsilon

        # define p for remainder on all feasible machines (same remaining)
        for m in M_i2[i_rem]:
            p_im2[(i_rem, m)] = float(remaining)

        # precedence: remainder depends on original i
        Pred_i2[i_rem] = [i]

        # beta for remainder (safe default 0 unless original was 1)
        beta_i2[i_rem] = 0

        # redirect successors: if k needs i, make it also need i_rem
        for k in I:
            if i in Pred_i2.get(k, []):
                if i_rem not in Pred_i2[k]:
                    Pred_i2[k].append(i_rem)

    data2 = dict(data)
    data2.update({
        "I": I2,
        "O_j": O_j2,
        "L_i": L_i2,
        "M_i": M_i2,
        "Pred_i": Pred_i2,
        "p_im": p_im2,
        "beta_i": beta_i2,
        "rem_map": rem_map
    })
    return data2


# ==========================================================
#  BASE DATA (same as your Step 4 file)
# ==========================================================

J = [1, 2, 3, 4, 5, 6, 7, 8]
I = list(range(1, 31))

O_j = {
    1: [1, 2, 3],
    2: [4, 5, 6, 7],
    3: [8, 9, 10, 11, 12],
    4: [13, 14],
    5: [15, 16, 17, 18],
    6: [19, 20, 21],
    7: [22, 23, 24, 25, 26],
    8: [27, 28, 29, 30],
}

M = list(range(1, 13))
machine_type = {1:1,2:1,3:1,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2,12:2}

K_i = {i: ([1] if (i % 2 == 1) else [2]) for i in I}
M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

L = list(range(1, 10))
L_i = {i: L[:] for i in I}
L_big = [1,2,3]
L_small = [4,5,6,7,8,9]

Pred_i = {i: [] for i in I}
Pred_i[2] = [1]
Pred_i[3] = [1]
Pred_i[5] = [4]
Pred_i[6] = [4]
Pred_i[7] = [5]
Pred_i[9] = [8]
Pred_i[10] = [8]
Pred_i[11] = [9,10]
Pred_i[12] = [11]
Pred_i[14] = [13]
Pred_i[16] = [15]
Pred_i[17] = [15]
Pred_i[18] = [16]
Pred_i[21] = [19,20]
Pred_i[23] = [22]
Pred_i[24] = [22]
Pred_i[25] = [23]
Pred_i[26] = [24]

# last depends on all in job
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for op in ops:
        if op != last:
            preds.add(op)
    Pred_i[last] = list(preds)

p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)
        machine_add = (m - 1) * 0.5
        p_im[(i, m)] = float(base + machine_add)

r_j = {}
release_times = [0.0,2.0,4.0,6.0,8.0,10.0,12.0,14.0]
for idx, j in enumerate(J):
    r_j[j] = release_times[idx]

d_i = {}
due_base = 30.0
Iend = [O_j[j][-1] for j in J]
for idx, i_last in enumerate(Iend):
    d_i[i_last] = due_base + 3.0 * idx
d_j = {j: d_i[O_j[j][-1]] for j in J}

g_j = {j: (1 if idx % 2 == 0 else 0) for idx, j in enumerate(J)}
p_j = {j: (1 if idx in [0,1,2] else 0) for idx, j in enumerate(J)}

t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

beta_i = {i: (1 if i in Iend else 0) for i in I}

M_proc = 1000.0
M_seq = 1000.0
M_Lseq = 1000.0


# ==========================================================
#  SIMPLE VISUALS
# ==========================================================

def plot_gantt_by_machine(I, M, M_i, x, S, C, title="Machine-wise schedule"):
    plt.figure()
    y_ticks, y_labels = [], []
    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        y_labels.append(f"Machine {m}")
        for i in I:
            if m not in M_i[i]:
                continue
            if (i, m) in x and x[i, m].X > 0.5:
                plt.barh(y_pos, C[i].X - S[i].X, left=S[i].X)
                plt.text(S[i].X, y_pos, f"{i}", va="center")
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


def plot_gantt_by_station(I, L, L_i, y, S, C, title="Station-wise schedule"):
    plt.figure()
    y_ticks, y_labels = [], []
    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)
        y_labels.append(f"Station {l}")
        for i in I:
            if l not in L_i[i]:
                continue
            if (i, l) in y and y[i, l].X > 0.5:
                plt.barh(y_pos, C[i].X - S[i].X, left=S[i].X)
                plt.text(S[i].X, y_pos, f"{i}", va="center")
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


# ==========================================================
#  BUILD MODEL
# ==========================================================

def build_model(name, data,
                unavailable_machines=None, unavailable_stations=None,
                freeze_done_ops=None,
                running_ops=None, mode="continue", t0=None,
                free_ops=None,
                old_solution=None):
    """
    mode:
      - "continue": freeze running ops fully
      - "optimize": keep[i] decides:
          keep=1 => freeze running op to old
          keep=0 => preempt: freeze done-part to [S_old, t0] and schedule remainder op (added to data)

    IMPORTANT: C_weld[j] is modeled as max completion over all ops in job (robust).
    """
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []
    freeze_done_ops = freeze_done_ops or []
    running_ops = running_ops or []
    free_ops = free_ops or []

    J = data["J"]; I = data["I"]; O_j = data["O_j"]
    M = data["M"]; L = data["L"]
    M_i = data["M_i"]; L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    r_j = data["r_j"]; d_j = data["d_j"]
    g_j = data["g_j"]; p_j = data["p_j"]
    t_grind_j = data["t_grind_j"]; t_paint_j = data["t_paint_j"]
    beta_i = data["beta_i"]
    rem_map = data.get("rem_map", {})

    model = Model(name)

    x = model.addVars([(i, m) for i in I for m in M_i[i]], vtype=GRB.BINARY, name="x")
    y = model.addVars([(i, l) for i in I for l in L_i[i]], vtype=GRB.BINARY, name="y")

    zM_index = [(i, h, m) for i in I for h in I if i != h for m in M]
    zM = model.addVars(zM_index, vtype=GRB.BINARY, name="zM")
    zL_index = [(i, h, l) for i in I for h in I if i != h for l in L]
    zL = model.addVars(zL_index, vtype=GRB.BINARY, name="zL")

    S = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="S")
    C = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    C_weld  = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="C_weld")
    C_final = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="C_final")

    T     = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="T")
    T_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T_max")
    C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

    model.setObjective(T_max, GRB.MINIMIZE)

    # assignment
    for i in I:
        model.addConstr(quicksum(x[i, m] for m in M_i[i]) == 1, name=f"assign_M_{i}")
        model.addConstr(quicksum(y[i, l] for l in L_i[i]) == 1, name=f"assign_L_{i}")

    # availability forbids
    if unavailable_machines:
        for (i, m) in x.keys():
            if m in unavailable_machines:
                model.addConstr(x[i, m] == 0, name=f"unavailM_{i}_{m}")
    if unavailable_stations:
        for (i, l) in y.keys():
            if l in unavailable_stations:
                model.addConstr(y[i, l] == 0, name=f"unavailL_{i}_{l}")

    # processing time
    for i in I:
        for m in M_i[i]:
            pijm = p_im[(i, m)]
            model.addConstr(C[i] - S[i] >= pijm - M_proc * (1 - x[i, m]), name=f"ptime_lb_{i}_{m}")
            model.addConstr(C[i] - S[i] <= pijm + M_proc * (1 - x[i, m]), name=f"ptime_ub_{i}_{m}")

    # big station feasibility
    for i in I:
        if beta_i.get(i, 0) == 1:
            for l in L_small:
                if l in L_i[i]:
                    model.addConstr(y[i, l] == 0, name=f"small_forbid_{i}_{l}")

    # precedence
    for i in I:
        for h in Pred_i.get(i, []):
            model.addConstr(S[i] >= C[h], name=f"prec_{h}_{i}")

    # release
    for j in J:
        first_op = O_j[j][0]
        model.addConstr(S[first_op] >= r_j[j], name=f"release_{j}")

    # machine sequencing
    I_list = I[:]
    nI = len(I_list)
    for idx_i in range(nI):
        for idx_h in range(idx_i + 1, nI):
            i = I_list[idx_i]
            h = I_list[idx_h]
            common_machines = set(M_i[i]).intersection(M_i[h])
            for m in common_machines:
                model.addConstr(
                    S[h] >= C[i]
                           - M_seq * (1 - zM[i, h, m])
                           - M_seq * (1 - x[i, m])
                           - M_seq * (1 - x[h, m]),
                    name=f"no_overlapM_h_after_i_{i}{h}{m}"
                )
                model.addConstr(
                    S[i] >= C[h]
                           - M_seq * zM[i, h, m]
                           - M_seq * (1 - x[i, m])
                           - M_seq * (1 - x[h, m]),
                    name=f"no_overlapM_i_after_h_{i}{h}{m}"
                )
                model.addConstr(zM[i, h, m] <= x[i, m], name=f"zM_le_xi_{i}{h}{m}")
                model.addConstr(zM[i, h, m] <= x[h, m], name=f"zM_le_xh_{i}{h}{m}")
                model.addConstr(zM[i, h, m] >= x[i, m] + x[h, m] - 1, name=f"zM_ge_sum_{i}{h}{m}")

    # station sequencing
    for idx_i in range(nI):
        for idx_h in range(idx_i + 1, nI):
            i = I_list[idx_i]
            h = I_list[idx_h]
            common_stations = set(L_i[i]).intersection(L_i[h])
            for l in common_stations:
                model.addConstr(
                    S[h] >= C[i]
                           - M_Lseq * (1 - zL[i, h, l])
                           - M_Lseq * (1 - y[i, l])
                           - M_Lseq * (1 - y[h, l]),
                    name=f"no_overlapL_h_after_i_{i}{h}{l}"
                )
                model.addConstr(
                    S[i] >= C[h]
                           - M_Lseq * zL[i, h, l]
                           - M_Lseq * (1 - y[i, l])
                           - M_Lseq * (1 - y[h, l]),
                    name=f"no_overlapL_i_after_h_{i}{h}{l}"
                )
                model.addConstr(zL[i, h, l] <= y[i, l], name=f"zL_le_yi_{i}{h}{l}")
                model.addConstr(zL[i, h, l] <= y[h, l], name=f"zL_le_yh_{i}{h}{l}")
                model.addConstr(zL[i, h, l] >= y[i, l] + y[h, l] - 1, name=f"zL_ge_sum_{i}{h}{l}")

    # ------------------------------------------------------
    # ROBUST Welding completion: max over all ops in job
    # ------------------------------------------------------
    for j in J:
        for op in O_j[j]:
            model.addConstr(C_weld[j] >= C[op], name=f"Cweld_ge_{j}_{op}")

    # Final completion
    for j in J:
        model.addConstr(
            C_final[j] == C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j],
            name=f"Cfinal_{j}"
        )

    # Tardiness, makespan
    for j in J:
        model.addConstr(T[j] >= C_final[j] - d_j[j], name=f"Tdef_{j}")
        model.addConstr(C_max >= C_final[j], name=f"Cmax_ge_{j}")
        model.addConstr(T_max >= T[j], name=f"Tmax_ge_{j}")

    # ------------------------------------------------------
    # STEP 4/5: timing/freeze rules around t0
    # ------------------------------------------------------

    # not-started ops cannot start before t0
    if t0 is not None and free_ops:
        for i in free_ops:
            if i in I:
                model.addConstr(S[i] >= t0, name=f"free_after_t0_{i}")

    keep = None
    if old_solution is not None:
        S_old = old_solution["S_old"]
        C_old = old_solution["C_old"]
        x_old = old_solution["x_old"]
        y_old = old_solution["y_old"]

        # freeze DONE always
        for i in freeze_done_ops:
            if i in I:
                model.addConstr(S[i] == S_old[i], name=f"freeze_done_S_{i}")
                model.addConstr(C[i] == C_old[i], name=f"freeze_done_C_{i}")
        for (i, m) in x.keys():
            if i in freeze_done_ops:
                model.addConstr(x[i, m] == x_old.get((i, m), 0), name=f"freeze_done_x_{i}_{m}")
        for (i, l) in y.keys():
            if i in freeze_done_ops:
                model.addConstr(y[i, l] == y_old.get((i, l), 0), name=f"freeze_done_y_{i}_{l}")

        # running handling
        if running_ops:
            if mode == "continue":
                # freeze all running
                for i in running_ops:
                    if i in I:
                        model.addConstr(S[i] == S_old[i], name=f"freeze_run_S_{i}")
                        model.addConstr(C[i] == C_old[i], name=f"freeze_run_C_{i}")
                for (i, m) in x.keys():
                    if i in running_ops:
                        model.addConstr(x[i, m] == x_old.get((i, m), 0), name=f"freeze_run_x_{i}_{m}")
                for (i, l) in y.keys():
                    if i in running_ops:
                        model.addConstr(y[i, l] == y_old.get((i, l), 0), name=f"freeze_run_y_{i}_{l}")

            else:
                # optimize: keep binary
                keep = model.addVars(running_ops, vtype=GRB.BINARY, name="keep_running")

                if t0 is None:
                    raise ValueError("t0 must be set when mode=optimize")

                for i in running_ops:
                    if i not in I:
                        continue

                    # keep=1 => 그대로 old schedule
                    model.addGenConstrIndicator(keep[i], True, S[i] == S_old[i], name=f"ind_keepS_{i}")
                    model.addGenConstrIndicator(keep[i], True, C[i] == C_old[i], name=f"ind_keepC_{i}")

                    # keep=0 => PREEMPT: done-part ends at t0 (freeze C[i]=t0) and remainder op handles rest
                    model.addGenConstrIndicator(keep[i], False, C[i] == t0, name=f"ind_preempt_Ceqt0_{i}")
                    # also ensure S[i] fixed to old (done-part started already)
                    model.addGenConstrIndicator(keep[i], False, S[i] == S_old[i], name=f"ind_preempt_SeqSold_{i}")

                    # freeze assignments for done-part when preempted too
                    for m in M_i[i]:
                        if (i, m) in x:
                            model.addGenConstrIndicator(keep[i], False, x[i, m] == x_old.get((i, m), 0),
                                                        name=f"ind_preempt_x_{i}_{m}")
                    for l in L_i[i]:
                        if (i, l) in y:
                            model.addGenConstrIndicator(keep[i], False, y[i, l] == y_old.get((i, l), 0),
                                                        name=f"ind_preempt_y_{i}_{l}")

                    # remainder op must exist for running ops (created in data rem_map)
                    i_rem = rem_map.get(i, None)
                    if i_rem is not None and i_rem in I:
                        # if keep=1 => remainder should be "off" (force it to have zero duration by fixing C=S=t0)
                        model.addGenConstrIndicator(keep[i], True, S[i_rem] == t0, name=f"ind_keep_remS_{i}")
                        model.addGenConstrIndicator(keep[i], True, C[i_rem] == t0, name=f"ind_keep_remC_{i}")

                        # if keep=0 => remainder starts after t0
                        model.addGenConstrIndicator(keep[i], False, S[i_rem] >= t0, name=f"ind_preempt_rem_after_{i}")

    model.update()
    return model, x, y, S, C, C_weld, C_final, T, T_max, C_max, keep


# ==========================================================
#  PACK BASE DATA
# ==========================================================

base_data = {
    "J": J, "I": I, "O_j": O_j,
    "M": M, "L": L,
    "machine_type": machine_type,
    "K_i": K_i, "M_i": M_i,
    "L_i": L_i,
    "Pred_i": Pred_i,
    "p_im": p_im,
    "r_j": r_j, "d_j": d_j,
    "g_j": g_j, "p_j": p_j,
    "t_grind_j": t_grind_j, "t_paint_j": t_paint_j,
    "beta_i": beta_i
}

# ==========================================================
#  BASELINE SOLVE
# ==========================================================

baseline, x, y, S, C, Cw, Cf, T, Tmax, Cmax, _ = build_model("Baseline", base_data)
baseline.optimize()

if baseline.SolCount == 0:
    print("No feasible baseline solution. Status =", baseline.Status)
    plt.show()
    raise SystemExit

print("\n===== BASELINE =====")
print(f"T_max = {Tmax.X:.2f} | C_max = {Cmax.X:.2f}")

S_old, C_old, x_old, y_old = extract_old_solution(I, x, y, S, C)
old_solution = {"S_old": S_old, "C_old": C_old, "x_old": x_old, "y_old": y_old}

plot_gantt_by_machine(I, M, M_i, x, S, C, title="BASELINE: Machine-wise schedule")
plot_gantt_by_station(I, L, L_i, y, S, C, title="BASELINE: Station-wise schedule")

# ==========================================================
#  RESCHEDULING STEP 5
# ==========================================================

print("\n==============================")
print("RESCHEDULING – STEP 5 (TRUE PREEMPTION)")
print("==============================")

t0 = compute_t0_hours("08:00")
print(f"Reschedule triggered NOW. t0 = {t0:.3f} hours since shift start")

I_done, I_run, I_free = classify_ops_by_t0(I, S_old, C_old, t0)
print(f"Done={len(I_done)} | Running={len(I_run)} | Not-started={len(I_free)}")

print("\nRUNNING ops handling:")
print("  1) devam (freeze running ops)")
print("  2) model karar versin (keep binary + TRUE SPLIT when keep=0)")
mode_in = input("Select [1/2] (default=1): ").strip()
mode = "continue" if mode_in in ["", "1"] else "optimize"
print(f"Selected mode = {mode}")

unavail_m = parse_int_list(input("Unavailable machines (e.g. 4,5) or empty: "))
unavail_l = parse_int_list(input("Unavailable stations (e.g. 2,9) or empty: "))

# urgent job
urgent_due_slack = 10.0
data1 = add_urgent_job(base_data, t0=t0, urgent_job_id=9, urgent_ops_count=3, urgent_due_slack=urgent_due_slack)

# STEP 5: add remainder ops for running
# (We create these remainders always; keep=1 will “turn them off”.)
data2 = add_split_remainders_for_running_ops(data1, I_run, t0, old_solution)

u = data2["urgent_job_id"]
u_ops = data2["urgent_ops"]
rem_map = data2.get("rem_map", {})

print(f"\nUrgent job: {u} ops={u_ops} | r_u={data2['r_j'][u]:.3f} d_u={data2['d_j'][u]:.3f}")
if I_run:
    print("Split remainders created for running ops:")
    for i in I_run:
        print(f"  running op {i} -> remainder op {rem_map.get(i)}")

resched, x2, y2, S2, C2, Cw2, Cf2, T2, Tmax2, Cmax2, keep = build_model(
    name="Reschedule_Step5",
    data=data2,
    unavailable_machines=unavail_m,
    unavailable_stations=unavail_l,
    freeze_done_ops=I_done,
    running_ops=I_run,
    mode=mode,
    t0=t0,
    free_ops=I_free,
    old_solution=old_solution
)

resched.optimize()

if resched.SolCount == 0:
    print("No feasible reschedule solution. Status =", resched.Status)
else:
    print("\n===== RESCHEDULE STEP 5 =====")
    print(f"T_max = {Tmax2.X:.2f} | C_max = {Cmax2.X:.2f}")

    print("\n===== URGENT JOB RESULT =====")
    print(f"Job {u}: C_final={Cf2[u].X:.2f}, T_u={T2[u].X:.2f}, d_u={data2['d_j'][u]:.2f}")

    if mode == "optimize" and keep is not None and I_run:
        print("\n===== RUNNING OPS DECISIONS =====")
        for i in I_run:
            k = int(round(keep[i].X))
            i_rem = rem_map.get(i)
            print(f"Op {i}: keep={k} | old({S_old[i]:.2f}->{C_old[i]:.2f}) | new({S2[i].X:.2f}->{C2[i].X:.2f})"
                  + (f" | rem op {i_rem}: ({S2[i_rem].X:.2f}->{C2[i_rem].X:.2f})" if i_rem in data2["I"] else ""))

    plot_gantt_by_machine(data2["I"], data2["M"], data2["M_i"], x2, S2, C2,
                          title="RESCHEDULE STEP 5: Machine-wise schedule")
    plot_gantt_by_station(data2["I"], data2["L"], data2["L_i"], y2, S2, C2,
                          title="RESCHEDULE STEP 5: Station-wise schedule")

plt.show()
