# -*- coding: utf-8 -*-
"""
Scenario: 4 jobs, 30 ops, bottleneck machine set for ops 1-25
Only provides data. No heuristic code here.
"""

def get_data():
    J = [1, 2, 3, 4]
    I = list(range(1, 31))

    O_j = {
        1: list(range(1, 16)),     # ops 1–15
        2: list(range(16, 26)),    # ops 16–25
        3: [26, 27, 28],
        4: [29, 30],
    }

    M = list(range(1, 13))   # 12 machines
    L = list(range(1, 10))   # 9 stations

    L_big = [1, 2, 3]
    L_small = [4, 5, 6, 7, 8, 9]

    # all stations feasible (big-station rule enforced by beta_i later)
    L_i = {i: L[:] for i in I}

    # Bottleneck machine feasibility
    BOTTLENECK_MACHINES = [1, 2]
    M_i = {}
    for i in I:
        if i <= 25:
            M_i[i] = BOTTLENECK_MACHINES[:]   # forced to 1–2
        else:
            mod = i % 4
            if mod == 1:
                M_i[i] = [1, 2, 3, 4, 5]
            elif mod == 2:
                M_i[i] = [2, 5, 6, 7, 8]
            elif mod == 3:
                M_i[i] = [1, 8, 9, 10]
            else:
                M_i[i] = [2, 9, 11, 12]

    # Precedence — chain per job + last depends on all previous
    Pred_i = {i: [] for i in I}
    for j in J:
        ops = O_j[j]
        for k in range(1, len(ops)):
            Pred_i[ops[k]].append(ops[k - 1])

        last = ops[-1]
        preds = set(Pred_i[last])
        for op in ops[:-1]:
            preds.add(op)
        Pred_i[last] = list(preds)

    # Processing times
    p_im = {}
    for i in I:
        for m in M_i[i]:
            base = 3 + (i % 5)
            machine_offset = (m - 1) * 0.15
            p_im[(i, m)] = float(base + machine_offset)

    # Release times
    r_j = {j: 0.0 for j in J}

    # Due dates
    d_j = {1: 100.0, 2: 110.0, 3: 120.0, 4: 130.0}

    # Grinding/Painting flags and times
    g_j = {1: 1, 2: 0, 3: 1, 4: 0}
    p_flag_j = {1: 1, 2: 1, 3: 0, 4: 0}

    t_grind_j = {}
    t_paint_j = {}
    for j in J:
        last_op = O_j[j][-1]
        t_grind_j[j] = 2.0 + (last_op % 2)
        t_paint_j[j] = 3.0 if p_flag_j[j] == 1 else 0.0

    # Big-station needs: last ops only
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
        "beta_i": beta_i
    }
