# -*- coding: utf-8 -*-
"""
Scenario: Welding Shop Scheduling data (4 jobs, 375 ops/job = 1500 ops)
Only provides data. No heuristic / plotting code here.

- Precedence: job içi lineer zincir (op -> op+1)
- Machines: 12 physical machines, 2 machine types (1=TIG, 2=MAG)
- Stations: 9 stations, big-station requirement for each job's last op
- Processing times: base = 3 + (i % 4)  (3–6), same for all machines
- Release times: all 0
- Due dates: intentionally short (50,60,70,80)
- Grinding/Painting flags and times as in original script
"""

def get_data():
    # ===============================
    #  DATA – 4 jobs, 375 ops/job
    # ===============================
    NUM_JOBS = 4
    OPS_PER_JOB = 375

    J = list(range(1, NUM_JOBS + 1))
    I = list(range(1, NUM_JOBS * OPS_PER_JOB + 1))  # 1..1500

    # JOB → OPERATIONS (each job is a consecutive block)
    O_j = {}
    cur = 1
    for j in J:
        ops_j = list(range(cur, cur + OPS_PER_JOB))
        O_j[j] = ops_j
        cur += OPS_PER_JOB

    # ===============================
    #  MACHINES & MACHINE TYPES
    # ===============================
    M = list(range(1, 13))  # 12 machines

    # machine types: 1=TIG, 2=MAG (as given)
    machine_type = {
        1: 1, 2: 1, 3: 1,          # TIG
        4: 2, 5: 2, 6: 2, 7: 2,
        8: 2, 9: 2, 10: 2, 11: 2, 12: 2
    }

    # Each operation can use both types => all machines feasible
    K_i = {i: [1, 2] for i in I}
    M_i = {
        i: [m for m in M if machine_type[m] in K_i[i]]
        for i in I
    }

    # ===============================
    #  STATIONS
    # ===============================
    L = list(range(1, 10))  # 9 stations
    L_i = {i: L[:] for i in I}

    L_big = [1, 2, 3]
    L_small = [4, 5, 6, 7, 8, 9]

    # ===============================
    #  PRECEDENCE – job içi lineer
    # ===============================
    Pred_i = {i: [] for i in I}
    for j in J:
        ops = O_j[j]
        for k in range(len(ops) - 1):
            i_from = ops[k]
            i_to = ops[k + 1]
            Pred_i[i_to].append(i_from)

    # ===============================
    #  PROCESSING TIMES p_im
    # ===============================
    p_im = {}
    for i in I:
        for m in M_i[i]:
            base = 3 + (i % 4)     # 3–6
            machine_factor = 0.0   # all equal speed
            p_im[(i, m)] = float(base + machine_factor)

    # ===============================
    #  RELEASE TIMES – job bazlı
    # ===============================
    r_j = {j: 0.0 for j in J}

    # ===============================
    #  DUE DATES – job bazlı (short)
    # ===============================
    d_j = {}
    base_due = 50.0
    for j in J:
        d_j[j] = base_due + 10.0 * (j - 1)  # 50,60,70,80

    # ===============================
    #  GRINDING & PAINTING
    # ===============================
    g_j = {}
    p_flag_j = {}
    for j in J:
        g_j[j] = 1 if j in [1, 3] else 0
        p_flag_j[j] = 1 if j in [1, 2] else 0

    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)              # 2 or 3
        t_paint_j[j] = 3.0 if p_flag_j[j] == 1 else 0.0

    # ===============================
    #  BIG-STATION REQUIREMENT
    # ===============================
    Iend = [O_j[j][-1] for j in J]  # last op per job
    beta_i = {i: 0 for i in I}
    for last in Iend:
        beta_i[last] = 1

    return {
        "J": J, "I": I, "O_j": O_j,
        "M": M, "L": L,
        "M_i": M_i, "L_i": L_i,
        "L_big": L_big, "L_small": L_small,
        "Pred_i": Pred_i,
        "p_im": p_im,
        "r_j": r_j, "d_j": d_j,
        "g_j": g_j, "p_flag_j": p_flag_j,
        "t_grind_j": t_grind_j, "t_paint_j": t_paint_j,
        "beta_i": beta_i,
        "machine_type": machine_type,
        "K_i": K_i
    }
