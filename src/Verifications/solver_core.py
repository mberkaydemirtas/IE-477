#!/usr/bin/env python3
# -- coding: utf-8 --

from gurobipy import Model, GRB, quicksum
from datetime import datetime, timedelta


# ==========================================================
#  TIME HELPERS
# ==========================================================

def compute_t0_hours(shift_start_hhmm="08:00"):
    """t0 = (now - shift_start) in HOURS"""
    now = datetime.now()
    hh, mm = map(int, shift_start_hhmm.split(":"))
    shift_start = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if now < shift_start:
        shift_start = shift_start - timedelta(days=1)
    return float((now - shift_start).total_seconds() / 3600.0)


# ==========================================================
#  SOLUTION EXTRACTION HELPERS
# ==========================================================

def extract_old_solution(I, x, y, S, C):
    """Store old schedule values from a solved model."""
    S_old = {int(i): float(S[i].X) for i in I}
    C_old = {int(i): float(C[i].X) for i in I}
    x_old = {(int(i), int(m)): int(round(x[i, m].X)) for (i, m) in x.keys()}
    y_old = {(int(i), int(l)): int(round(y[i, l].X)) for (i, l) in y.keys()}
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


def op_to_job_map(O_j):
    """op_id -> job_id"""
    mp = {}
    for j, ops in O_j.items():
        for i in ops:
            mp[int(i)] = int(j)
    return mp


def extract_assignment_for_ops(I, M_i, L_i, x, y):
    """For each op i, return chosen machine and station from solved vars."""
    chosen_m = {}
    chosen_l = {}

    for i in I:
        mm = None
        for m in M_i[i]:
            if (i, m) in x and x[i, m].X > 0.5:
                mm = int(m)
                break
        ll = None
        for l in L_i[i]:
            if (i, l) in y and y[i, l].X > 0.5:
                ll = int(l)
                break

        chosen_m[int(i)] = mm
        chosen_l[int(i)] = ll

    return chosen_m, chosen_l


# ==========================================================
#  DATA BUILD (your current synthetic dataset)
# ==========================================================

def make_base_data():
    # Jobs
    J = [1, 2, 3, 4, 5, 6, 7, 8]

    # Operations
    I = list(range(1, 31))

    # Job -> operations mapping
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

    # Machines
    M = list(range(1, 13))
    machine_type = {
        1: 1, 2: 1, 3: 1,   # TIG
        4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2  # MAG
    }

    # Odd ops -> TIG(1), Even ops -> MAG(2)
    K_i = {i: ([1] if (i % 2 == 1) else [2]) for i in I}
    M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

    # Stations
    L = list(range(1, 10))
    L_i = {i: L[:] for i in I}
    L_big = [1, 2, 3]
    L_small = [4, 5, 6, 7, 8, 9]

    # Precedence
    Pred_i = {i: [] for i in I}
    Pred_i[2]  = [1]
    Pred_i[3]  = [1]
    Pred_i[5]  = [4]
    Pred_i[6]  = [4]
    Pred_i[7]  = [5]
    Pred_i[9]  = [8]
    Pred_i[10] = [8]
    Pred_i[11] = [9, 10]
    Pred_i[12] = [11]
    Pred_i[14] = [13]
    Pred_i[16] = [15]
    Pred_i[17] = [15]
    Pred_i[18] = [16]
    Pred_i[21] = [19, 20]
    Pred_i[23] = [22]
    Pred_i[24] = [22]
    Pred_i[25] = [23]
    Pred_i[26] = [24]

    # "last depends on all in job"
    for j in J:
        ops = O_j[j]
        last = ops[-1]
        preds = set(Pred_i[last])
        for op in ops:
            if op != last:
                preds.add(op)
        Pred_i[last] = list(preds)

    # Processing times
    p_im = {}
    for i in I:
        for m in M_i[i]:
            base = 3 + (i % 5)
            machine_add = (m - 1) * 0.5
            p_im[(i, m)] = float(base + machine_add)

    # Releases
    release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    r_j = {j: release_times[idx] for idx, j in enumerate(J)}

    # Due dates
    d_i = {}
    due_base = 30.0
    Iend = [O_j[j][-1] for j in J]
    for idx, i_last in enumerate(Iend):
        d_i[i_last] = due_base + 3.0 * idx
    d_j = {j: d_i[O_j[j][-1]] for j in J}

    # grind/paint flags and times
    g_j = {j: (1 if idx % 2 == 0 else 0) for idx, j in enumerate(J)}
    p_j = {j: (1 if idx in [0, 1, 2] else 0) for idx, j in enumerate(J)}

    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)
        t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

    # beta (last ops require big station)
    beta_i = {i: (1 if i in Iend else 0) for i in I}

    # Big-M
    M_proc = 1000.0
    M_seq = 1000.0
    M_Lseq = 1000.0

    return {
        "J": J, "I": I, "O_j": O_j,
        "M": M, "L": L,
        "machine_type": machine_type,
        "K_i": K_i,
        "M_i": M_i,
        "L_i": L_i,
        "L_big": L_big, "L_small": L_small,
        "Pred_i": Pred_i,
        "p_im": p_im,
        "r_j": r_j, "d_j": d_j,
        "g_j": g_j, "p_j": p_j,
        "t_grind_j": t_grind_j, "t_paint_j": t_paint_j,
        "beta_i": beta_i,
        "M_proc": M_proc, "M_seq": M_seq, "M_Lseq": M_Lseq
    }


# ==========================================================
#  URGENT JOB EXTENSION
# ==========================================================

def add_urgent_job(data, t0, urgent_job_id=9, urgent_ops_count=3, urgent_due_slack=10.0):
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

    L_i2 = dict(L_i)
    for i in new_ops:
        L_i2[i] = L[:]

    K_i2 = dict(K_i)
    M_i2 = dict(M_i)
    for i in new_ops:
        K_i2[i] = [1] if (i % 2 == 1) else [2]
        M_i2[i] = [m for m in M if machine_type[m] in K_i2[i]]

    p_im2 = dict(p_im)
    for i in new_ops:
        for m in M_i2[i]:
            base = 3 + (i % 5)
            machine_add = (m - 1) * 0.5
            p_im2[(i, m)] = float(base + machine_add)

    Pred_i2 = {i: list(Pred_i[i]) for i in Pred_i}
    for i in new_ops:
        Pred_i2[i] = []

    # chain
    for k in range(1, len(new_ops)):
        Pred_i2[new_ops[k]] = [new_ops[k - 1]]

    # last depends on all
    last = new_ops[-1]
    preds = set(Pred_i2[last])
    for op in new_ops:
        if op != last:
            preds.add(op)
    Pred_i2[last] = list(preds)

    r_j2 = dict(r_j)
    d_j2 = dict(d_j)
    r_j2[urgent_job_id] = float(t0)
    d_j2[urgent_job_id] = float(t0 + urgent_due_slack)

    g_j2 = dict(g_j)
    p_j2 = dict(p_j)
    t_grind_j2 = dict(t_grind_j)
    t_paint_j2 = dict(t_paint_j)

    g_j2[urgent_job_id] = 0
    p_j2[urgent_job_id] = 0
    t_grind_j2[urgent_job_id] = 0.0
    t_paint_j2[urgent_job_id] = 0.0

    beta_i2 = dict(beta_i)
    for i in new_ops:
        beta_i2[i] = 0
    beta_i2[last] = 1

    out = dict(data)
    out.update({
        "J": J2, "I": I2, "O_j": O_j2,
        "K_i": K_i2, "M_i": M_i2, "L_i": L_i2,
        "Pred_i": Pred_i2, "p_im": p_im2,
        "r_j": r_j2, "d_j": d_j2,
        "g_j": g_j2, "p_j": p_j2,
        "t_grind_j": t_grind_j2, "t_paint_j": t_paint_j2,
        "beta_i": beta_i2,
        "urgent_job_id": urgent_job_id,
        "urgent_ops": new_ops
    })
    return out


# ==========================================================
#  STEP 5: SPLIT REMAINDERS FOR RUNNING OPS
# ==========================================================

def add_split_remainders_for_running_ops(data, I_run, t0, old_solution):
    """
    For each running op i, create a remainder op i_rem with:
      - p(i_rem) = remaining time (based on old assigned machine)
      - precedence: i -> i_rem
      - and i_rem becomes predecessor of successors of i
    """
    J, I, O_j = data["J"], data["I"], data["O_j"]
    M_i = data["M_i"]
    L_i = data["L_i"]
    Pred_i = data["Pred_i"]
    p_im = data["p_im"]
    beta_i = data["beta_i"]

    S_old = old_solution["S_old"]
    x_old = old_solution["x_old"]

    next_op = max(I) + 1
    rem_map = {}

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

    def old_machine(i):
        for (ii, m), v in x_old.items():
            if ii == i and v == 1:
                return m
        return None

    for i in I_run:
        i_rem = next_op
        next_op += 1
        rem_map[i] = i_rem

        I2.append(i_rem)

        # Attach remainder to same job (for reporting)
        j = op_to_job.get(i, None)
        if j is not None:
            O_j2[j].append(i_rem)

        L_i2[i_rem] = list(L_i[i])
        M_i2[i_rem] = list(M_i[i])

        m0 = old_machine(i)
        if m0 is None:
            m0 = M_i[i][0]

        full_p = p_im[(i, m0)]
        elapsed = max(0.0, t0 - S_old[i])
        remaining = max(0.01, full_p - elapsed)

        for m in M_i2[i_rem]:
            p_im2[(i_rem, m)] = float(remaining)

        # precedence links
        Pred_i2[i_rem] = [i]
        beta_i2[i_rem] = 0

        # if i was predecessor of some k, add i_rem as predecessor too
        for k in I:
            if i in Pred_i2.get(k, []):
                if i_rem not in Pred_i2[k]:
                    Pred_i2[k].append(i_rem)

    out = dict(data)
    out.update({
        "I": I2,
        "O_j": O_j2,
        "L_i": L_i2,
        "M_i": M_i2,
        "Pred_i": Pred_i2,
        "p_im": p_im2,
        "beta_i": beta_i2,
        "rem_map": rem_map
    })
    return out


# ==========================================================
#  MODEL BUILDER
# ==========================================================

def build_model(name, data,
                unavailable_machines=None, unavailable_stations=None,
                freeze_done_ops=None,
                running_ops=None, mode="continue", t0=None,
                free_ops=None,
                old_solution=None):
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
    L_small = data["L_small"]
    M_proc = data["M_proc"]; M_seq = data["M_seq"]; M_Lseq = data["M_Lseq"]
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

    # Objective unchanged
    model.setObjective(T_max, GRB.MINIMIZE)

    # assignments
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

    # processing time linking
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
    for a in range(nI):
        for b in range(a + 1, nI):
            i = I_list[a]
            h = I_list[b]
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
                model.addConstr(zM[i, h, m] <= x[i, m])
                model.addConstr(zM[i, h, m] <= x[h, m])
                model.addConstr(zM[i, h, m] >= x[i, m] + x[h, m] - 1)

    # station sequencing
    for a in range(nI):
        for b in range(a + 1, nI):
            i = I_list[a]
            h = I_list[b]
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
                model.addConstr(zL[i, h, l] <= y[i, l])
                model.addConstr(zL[i, h, l] <= y[h, l])
                model.addConstr(zL[i, h, l] >= y[i, l] + y[h, l] - 1)

    # robust welding completion: max over all job ops
    for j in J:
        for op in O_j[j]:
            model.addConstr(C_weld[j] >= C[op], name=f"Cweld_ge_{j}_{op}")

    # final completion
    for j in J:
        model.addConstr(
            C_final[j] == C_weld[j] + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j],
            name=f"Cfinal_{j}"
        )

    # tardiness, makespan, Tmax
    for j in J:
        model.addConstr(T[j] >= C_final[j] - d_j[j], name=f"Tdef_{j}")
        model.addConstr(C_max >= C_final[j], name=f"Cmax_ge_{j}")
        model.addConstr(T_max >= T[j], name=f"Tmax_ge_{j}")

    # rescheduling rule: free ops cannot start before t0
    if t0 is not None and free_ops:
        for i in free_ops:
            if i in I:
                model.addConstr(S[i] >= t0, name=f"free_after_t0_{i}")

    # Freeze framework
    keep = None
    if old_solution is not None:
        S_old = old_solution["S_old"]
        C_old = old_solution["C_old"]
        x_old = old_solution["x_old"]
        y_old = old_solution["y_old"]

        # freeze done
        for i in freeze_done_ops:
            if i in I:
                model.addConstr(S[i] == S_old[i], name=f"freeze_done_S_{i}")
                model.addConstr(C[i] == C_old[i], name=f"freeze_done_C_{i}")
        for (i, m) in x.keys():
            if i in freeze_done_ops:
                model.addConstr(x[i, m] == x_old.get((i, m), 0))
        for (i, l) in y.keys():
            if i in freeze_done_ops:
                model.addConstr(y[i, l] == y_old.get((i, l), 0))

        # running handling
        if running_ops:
            if mode == "continue":
                # freeze running ops
                for i in running_ops:
                    if i in I:
                        model.addConstr(S[i] == S_old[i], name=f"freeze_run_S_{i}")
                        model.addConstr(C[i] == C_old[i], name=f"freeze_run_C_{i}")
                for (i, m) in x.keys():
                    if i in running_ops:
                        model.addConstr(x[i, m] == x_old.get((i, m), 0))
                for (i, l) in y.keys():
                    if i in running_ops:
                        model.addConstr(y[i, l] == y_old.get((i, l), 0))

            else:
                # mode == optimize
                if t0 is None:
                    raise ValueError("t0 must be set for mode=optimize")

                keep = model.addVars(running_ops, vtype=GRB.BINARY, name="keep_running")

                for i in running_ops:
                    if i not in I:
                        continue

                    # keep=1 => continue old schedule
                    model.addGenConstrIndicator(keep[i], True, S[i] == S_old[i], name=f"ind_keepS_{i}")
                    model.addGenConstrIndicator(keep[i], True, C[i] == C_old[i], name=f"ind_keepC_{i}")

                    # keep=0 => preempt: done part ends at t0
                    model.addGenConstrIndicator(keep[i], False, S[i] == S_old[i], name=f"ind_preemptS_{i}")
                    model.addGenConstrIndicator(keep[i], False, C[i] == t0,      name=f"ind_preemptC_{i}")

                    # freeze assignments when preempted too
                    for m in M_i[i]:
                        if (i, m) in x:
                            model.addGenConstrIndicator(keep[i], False, x[i, m] == x_old.get((i, m), 0))
                    for l in L_i[i]:
                        if (i, l) in y:
                            model.addGenConstrIndicator(keep[i], False, y[i, l] == y_old.get((i, l), 0))

                    # remainder operation
                    i_rem = rem_map.get(i, None)
                    if i_rem is not None and i_rem in I:
                        # keep=1 => remainder forced to 0-length at t0 (inactive)
                        model.addGenConstrIndicator(keep[i], True, S[i_rem] == t0)
                        model.addGenConstrIndicator(keep[i], True, C[i_rem] == t0)
                        # keep=0 => remainder starts after t0
                        model.addGenConstrIndicator(keep[i], False, S[i_rem] >= t0)

    model.update()
    return model, x, y, S, C, C_weld, C_final, T, T_max, C_max, keep


# ==========================================================
#  SOLVE WRAPPERS
# ==========================================================

def solve_baseline(data):
    model, x, y, S, C, Cw, Cf, T, Tmax, Cmax, _ = build_model("Baseline", data)
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"No feasible baseline. Status={model.Status}")

    I = data["I"]
    S_old, C_old, x_old, y_old = extract_old_solution(I, x, y, S, C)

    job_of = op_to_job_map(data["O_j"])
    chosen_m, chosen_l = extract_assignment_for_ops(I, data["M_i"], data["L_i"], x, y)

    schedule = []
    for i in I:
        schedule.append({
            "op_id": int(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(S[i].X),
            "finish": float(C[i].X),
            "machine": chosen_m[int(i)],
            "station": chosen_l[int(i)],
        })

    return {
        "objective": {"T_max": float(Tmax.X), "C_max": float(Cmax.X)},
        "schedule": schedule,
        "S_old": S_old, "C_old": C_old,
        "x_old": x_old, "y_old": y_old
    }


def solve_reschedule(data_base,
                    old_solution,
                    shift_start_hhmm="08:00",
                    unavailable_machines=None,
                    unavailable_stations=None,
                    mode="continue",
                    urgent_job_id=9,
                    urgent_ops_count=3,
                    urgent_due_slack=10.0):
    unavailable_machines = unavailable_machines or []
    unavailable_stations = unavailable_stations or []

    # trigger time
    t0 = compute_t0_hours(shift_start_hhmm)

    # classify according to OLD solution times
    I = data_base["I"]
    I_done, I_run, I_free = classify_ops_by_t0(I, old_solution["S_old"], old_solution["C_old"], t0)

    # add urgent + add remainders
    data1 = add_urgent_job(data_base, t0, urgent_job_id, urgent_ops_count, urgent_due_slack)
    data2 = add_split_remainders_for_running_ops(data1, I_run, t0, old_solution)

    # build & solve
    model, x, y, S, C, Cw, Cf, T, Tmax, Cmax, keep = build_model(
        name="Reschedule",
        data=data2,
        unavailable_machines=unavailable_machines,
        unavailable_stations=unavailable_stations,
        freeze_done_ops=I_done,
        running_ops=I_run,
        mode=mode,
        t0=t0,
        free_ops=I_free,
        old_solution=old_solution
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError(f"No feasible reschedule. Status={model.Status}")

    urgent = data2.get("urgent_job_id", urgent_job_id)
    urgent_ops = data2.get("urgent_ops", [])

    keep_out = {}
    if mode == "optimize" and keep is not None:
        for i in I_run:
            keep_out[int(i)] = int(round(keep[i].X))

    # UI-ready schedule list
    job_of = op_to_job_map(data2["O_j"])
    chosen_m, chosen_l = extract_assignment_for_ops(data2["I"], data2["M_i"], data2["L_i"], x, y)

    schedule_new = []
    for i in data2["I"]:
        schedule_new.append({
            "op_id": int(i),
            "job_id": int(job_of.get(i, -1)),
            "start": float(S[i].X),
            "finish": float(C[i].X),
            "machine": chosen_m[int(i)],
            "station": chosen_l[int(i)],
        })

    # optional: diff for original ops
    changed_ops = []
    for i in data_base["I"]:
        old_s = old_solution["S_old"].get(int(i), None)
        old_c = old_solution["C_old"].get(int(i), None)
        if old_s is None:
            continue

        new_s = float(S[i].X) if i in S else None
        new_c = float(C[i].X) if i in C else None
        if new_s is None or new_c is None:
            continue

        if abs(new_s - old_s) > 1e-6 or abs(new_c - old_c) > 1e-6:
            changed_ops.append({
                "op_id": int(i),
                "old_start": float(old_s), "old_finish": float(old_c),
                "new_start": float(new_s), "new_finish": float(new_c),
            })

    return {
        "t0": float(t0),
        "sets": {"I_done": I_done, "I_run": I_run, "I_free": I_free},
        "objective": {"T_max": float(Tmax.X), "C_max": float(Cmax.X)},
        "urgent": {
            "job_id": int(urgent),
            "ops": [int(i) for i in urgent_ops],
            "C_final": float(Cf[urgent].X),
            "T": float(T[urgent].X),
            "d": float(data2["d_j"][urgent])
        },
        "keep_decisions": keep_out,
        "schedule": schedule_new,
        "changed_ops": changed_ops
    }
