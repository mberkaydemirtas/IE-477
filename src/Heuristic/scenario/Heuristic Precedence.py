# -*- coding: utf-8 -*-
"""
Scenario: HARD PRECEDENCE TEST (2 jobs, 6 ops) — DATA ONLY
Converted to standard get_data() format.

IMPORTANT: Data preserved from the original heuristic visualization script.
- p_j renamed to p_flag_j for compatibility.
- K, machine_type, K_i kept for traceability (heuristic can ignore).
"""

def get_data():
    # ===============================
    # JOBS & OPERATIONS
    # ===============================
    J = [1, 2]
    I = [1, 2, 3, 4, 5, 6]

    O_j = {
        1: [1, 2, 3, 4],
        2: [5, 6],
    }

    # ===============================
    # MACHINES + TYPES
    # ===============================
    M = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    K = [1, 2]  # 1=TIG, 2=MAG

    machine_type = {
        1: 1, 2: 1, 3: 2, 4: 2,
        5: 1, 6: 1, 7: 1, 8: 2,
        9: 1, 10: 1, 11: 1, 12: 1,
    }

    # Feasible machine types per op (pattern by i % 3)
    K_i = {}
    for i in I:
        r = i % 3
        if r == 1:
            K_i[i] = [1]
        elif r == 2:
            K_i[i] = [2]
        else:
            K_i[i] = [1, 2]

    # Feasible machines per op
    M_i = {i: [m for m in M if machine_type[m] in K_i[i]] for i in I}

    # ===============================
    # STATIONS
    # ===============================
    L = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    L_i = {i: L[:] for i in I}

    L_big   = [1, 2, 3]
    L_small = [4, 5, 6, 7, 8, 9]

    # ===============================
    # PRECEDENCE
    # ===============================
    Pred_i = {i: [] for i in I}

    # Job 1: 1 -> 2, 1 -> 3, (2,3) -> 4
    Pred_i[2] = [1]
    Pred_i[3] = [1]
    Pred_i[4] = [2, 3]

    # Job 2: 5 -> 6
    Pred_i[6] = [5]

    # Cross-job: 4 -> 5
    Pred_i[5] = [4]

    # Optional step kept from original: each job's last op depends on all ops in that job
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
    base_p = {1: 2.0, 2: 4.0, 3: 4.0, 4: 3.0, 5: 5.0, 6: 2.0}
    p_im = {}
    for i in I:
        for m in M_i[i]:
            p_im[(i, m)] = float(base_p[i] + 0.2 * (m - 1))

    # ===============================
    # RELEASE / DUE
    # ===============================
    r_j = {1: 0.0, 2: 0.0}
    d_j = {1: 11.0, 2: 16.0}

    # ===============================
    # GRINDING / PAINTING
    # ===============================
    g_j = {1: 1, 2: 0}
    p_flag_j = {1: 0, 2: 0}  # was p_j

    # Times kept exactly as original
    t_grind_j = {1: 2.0, 2: 0.0}
    t_paint_j = {1: 0.0, 2: 0.0}

    # ===============================
    # BIG-STATION REQUIREMENT beta_i
    # ===============================
    # only job-end ops require big station
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
        # optional traceability:
        "K": K, "machine_type": machine_type, "K_i": K_i
    }
