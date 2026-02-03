# -*- coding: utf-8 -*-
"""
Scenario: One Very Urgent Job (Job 1 has 10 operations) — DATA ONLY
Converted to standard get_data() format.

IMPORTANT: Data preserved exactly from the provided script.
- p_j renamed to p_flag_j for compatibility with heuristic interface.
"""

def get_data():
    # ===============================
    # JOBS & OPERATIONS
    # ===============================
    J = [1, 2, 3, 4, 5, 6, 7, 8]
    I = list(range(1, 31))  # 1..30

    O_j = {
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        2: [11, 12, 13, 14],
        3: [15, 16, 17, 18],
        4: [19, 20],
        5: [21, 22, 23],
        6: [24, 25],
        7: [26, 27, 28],
        8: [29, 30],
    }

    # ===============================
    # MACHINES + TYPES
    # ===============================
    M = [1, 2, 3, 4]
    K = [1, 2]  # 1=TIG, 2=MAG (traceability)

    machine_type = {
        1: 1,  # TIG
        2: 1,  # TIG
        3: 2,  # MAG
        4: 2,  # MAG
    }

    # Feasible machine types per op (pattern by i % 3)
    K_i = {}
    for i in I:
        r = i % 3
        if r == 1:
            K_i[i] = [1]        # only TIG
        elif r == 2:
            K_i[i] = [2]        # only MAG
        else:
            K_i[i] = [1, 2]     # both types

    # Feasible machines per operation
    M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

    # ===============================
    # STATIONS
    # ===============================
    L = [1, 2, 3, 4]
    L_i = {i: [1, 2, 3, 4] for i in I}

    L_big   = [1, 2, 3]
    L_small = [4]

    # ===============================
    # PRECEDENCE (linear per job + last depends on all previous)
    # ===============================
    Pred_i = {i: [] for i in I}

    # linear chain inside each job
    for j in J:
        ops = O_j[j]
        for k in range(1, len(ops)):
            Pred_i[ops[k]].append(ops[k - 1])

    # last op depends on all previous ops in that job
    for j in J:
        ops = O_j[j]
        last = ops[-1]
        preds = set(Pred_i[last])
        for op in ops:
            if op != last:
                preds.add(op)
        Pred_i[last] = list(preds)

    # ===============================
    # PROCESSING TIMES
    # ===============================
    p_im = {}
    for i in I:
        for m in M_i[i]:
            base = 3 + (i % 5)          # 3..7
            machine_add = (m - 1) * 0.5
            p_im[(i, m)] = float(base + machine_add)

    # ===============================
    # RELEASE TIMES
    # ===============================
    r_j = {j: 0.0 for j in J}

    # ===============================
    # DUE DATES (Job 1 very urgent)
    # ===============================
    d_j = {
        1: 25.0,
        2: 80.0,
        3: 90.0,
        4: 85.0,
        5: 100.0,
        6: 95.0,
        7: 110.0,
        8: 120.0,
    }

    # ===============================
    # GRINDING / PAINTING
    # ===============================
    g_j = {
        1: 1,
        2: 0,
        3: 1,
        4: 0,
        5: 1,
        6: 0,
        7: 1,
        8: 0,
    }

    # Original variable p_j -> rename to p_flag_j
    p_flag_j = {
        1: 1,
        2: 1,
        3: 1,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
    }

    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)          # 2 or 3
        t_paint_j[j] = 3.0 if p_flag_j[j] == 1 else 0.0

    # ===============================
    # BIG-STATION REQUIREMENT beta_i
    # ===============================
    # only each job's last operation requires big station
    Iend = [O_j[j][-1] for j in J]
    beta_i = {i: (1 if i in Iend else 0) for i in I}

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
        # optional traceability fields:
        "K": K, "machine_type": machine_type, "K_i": K_i
    }
