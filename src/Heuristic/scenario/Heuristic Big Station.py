# -*- coding: utf-8 -*-
"""
Scenario: LARGE EXAMPLE (8 jobs, 30 ops) — DATA ONLY
Converted to the standard get_data() format (no model/solver code).

IMPORTANT: Values are kept identical to the original script:
- machine types -> K_i -> M_i
- stations, L_big/L_small, L_i
- precedence including "last op depends on all previous ops in the job"
- heavier processing times p_im
- release times r_j
- due dates derived from last ops (d_i -> d_j)
- grinding g_j, painting p_flag_j (was p_j), grind/paint times
- beta_i: most ops require big station; only first op of each job is flexible
"""

def get_data():
    # ===============================
    # JOBS & OPERATIONS
    # ===============================
    J = [1, 2, 3, 4, 5, 6, 7, 8]
    I = list(range(1, 31))  # 1..30

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

    # ===============================
    # MACHINES (12) + TYPES (K)
    # ===============================
    M = list(range(1, 13))  # 1..12
    K = [1, 2]  # 1=TIG, 2=MAG (kept for traceability, not required by heuristic)

    machine_type = {
        1: 1,  # TIG
        2: 1,  # TIG
        3: 1,  # TIG
        4: 2,  # MAG
        5: 2,  # MAG
        6: 2,  # MAG
        7: 2,  # MAG
        8: 2,  # MAG
        9: 2,  # MAG
        10: 2, # MAG
        11: 2, # MAG
        12: 2, # MAG
    }

    # For each operation: feasible machine TYPES K_i
    # (pattern: odd ops TIG, even ops MAG)
    K_i = {}
    for i in I:
        if i % 2 == 1:
            K_i[i] = [1]  # TIG
        else:
            K_i[i] = [2]  # MAG

    # Feasible machine sets M_i induced by types
    M_i = {
        i: [m for m in M if machine_type[m] in K_i[i]]
        for i in I
    }

    # ===============================
    # STATIONS
    # ===============================
    L = [1, 2, 3, 4]

    # Every operation can go to any station (size rule enforced by beta_i)
    L_i = {i: [1, 2, 3, 4] for i in I}

    # Only station 1 is big, 2–4 are small
    L_big   = [1]
    L_small = [2, 3, 4]

    # ===============================
    # PRECEDENCE
    # ===============================
    Pred_i = {i: [] for i in I}

    # Job 1: 1 → 2, 1 → 3
    Pred_i[2] = [1]
    Pred_i[3] = [1]

    # Job 2: 4 → 5, 4 → 6, 5 → 7
    Pred_i[5] = [4]
    Pred_i[6] = [4]
    Pred_i[7] = [5]

    # Job 3: 8 → 9, 8 → 10, 9 & 10 → 11, 11 → 12
    Pred_i[9]  = [8]
    Pred_i[10] = [8]
    Pred_i[11] = [9, 10]
    Pred_i[12] = [11]

    # Job 4: 13 → 14
    Pred_i[14] = [13]

    # Job 5: 15 → 16, 15 → 17, 16 → 18
    Pred_i[16] = [15]
    Pred_i[17] = [15]
    Pred_i[18] = [16]

    # Job 6: 19 & 20 → 21
    Pred_i[21] = [19, 20]

    # Job 7: 22 → 23, 22 → 24, 23 → 25, 24 → 26
    Pred_i[23] = [22]
    Pred_i[24] = [22]
    Pred_i[25] = [23]
    Pred_i[26] = [24]

    # Job 8: [27, 28, 29, 30] no internal precedence initially

    # Extra step: each job's last operation must come after ALL ops in that job
    for j in J:
        ops = O_j[j]
        last = ops[-1]
        preds = set(Pred_i[last])
        for i_op in ops:
            if i_op == last:
                continue
            preds.add(i_op)
        Pred_i[last] = list(preds)

    # ===============================
    # PROCESSING TIMES (HEAVIER)
    # ===============================
    p_im = {}
    for i in I:
        for m in M_i[i]:
            base = 6 + (i % 7)          # 6..12
            machine_add = (m - 1) * 0.8 # bigger spread
            p_im[(i, m)] = float(base + machine_add)

    # ===============================
    # RELEASE TIMES
    # ===============================
    r_j = {}
    release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    for idx, j in enumerate(J):
        r_j[j] = release_times[idx]

    # ===============================
    # DUE DATES (TIGHT) — via last ops
    # ===============================
    d_i = {}
    due_base = 22.0
    Iend = [O_j[j][-1] for j in J]
    for idx, i_last in enumerate(Iend):
        d_i[i_last] = due_base + 1.0 * idx  # 22..29

    d_j = {}
    for j in J:
        last_op = O_j[j][-1]
        d_j[j] = d_i[last_op]

    # ===============================
    # GRINDING / PAINTING
    # ===============================
    # Grinding requirement g_j
    g_j = {}
    for idx, j in enumerate(J):
        g_j[j] = 1 if idx % 2 == 0 else 0  # j=1,3,5,7 grind

    # Painting requirement (original variable was p_j)
    p_flag_j = {}
    for idx, j in enumerate(J):
        p_flag_j[j] = 1 if idx in [0, 1, 2] else 0  # j=1,2,3 paint

    # Grinding and painting times
    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)
        t_paint_j[j] = 3.0 if p_flag_j[j] == 1 else 0.0

    # ===============================
    # BIG-STATION REQUIREMENT beta_i
    # ===============================
    # Only the FIRST operation of each job is flexible (beta=0).
    # All others MUST use big station (beta=1).
    beta_i = {}
    first_ops = [O_j[j][0] for j in J]  # [1,4,8,13,15,19,22,27]
    for i in I:
        beta_i[i] = 0 if i in first_ops else 1

    # Return in the exact dictionary style you want
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
        # optional traceability fields (harmless if heuristic ignores):
        "K": K, "machine_type": machine_type, "K_i": K_i
    }
