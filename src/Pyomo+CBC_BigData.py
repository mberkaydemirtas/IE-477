#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Welding Shop Scheduling – PYOMO
Veri seti: 4 jobs, 80 ops, 12 machines, 9 stations
Makine tipleri (TIG / MAG) + big/small station + precedence
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import matplotlib.pyplot as plt
from check import check_solution


# ===============================  
#  DATA – 4 jobs, 80 ops
# ===============================

# Jobs
J = [1, 2, 3, 4]

# Operations (80 ops total)
I = list(range(1, 81))  # 1..80

# JOB → OPERATIONS
# Job 1: 18 op
# Job 2: 22 op
# Job 3: 19 op
# Job 4: 21 op   (toplam 80)
O_j = {
    1: list(range(1, 19)),     # 1–18
    2: list(range(19, 41)),    # 19–40
    3: list(range(41, 60)),    # 41–59
    4: list(range(60, 81)),    # 60–80
}

# ===============================
#  MACHINES & MACHINE TYPES
# ===============================

# Fiziksel makineler (12 adet)
M = list(range(1, 13))  # [1,2,...,12]

# Makine tipleri (2 tip)
# 1 = TIG welding
# 2 = MAG welding
K = [1, 2]

# Her makinenin tipi
# Makine 1-3: TIG
# Makine 4-12: MAG
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

# Her operasyon için UYGUN makine tipleri (K_i)
# ÖRNEK PATTERN:
# - Tek sayılı operasyonlar: sadece TIG (tip 1)
# - Çift sayılı operasyonlar: sadece MAG (tip 2)
K_i = {}
for i in I:
    if i % 2 == 1:
        K_i[i] = [1]      # TIG
    else:
        K_i[i] = [2]      # MAG

# FEASIBLE MACHINE SETS M_i
# M_i[i] = { m ∈ M : machine_type[m] ∈ K_i[i] }
M_i = {
    i: [m for m in M if machine_type[m] in K_i[i]]
    for i in I
}

# ===============================
#  STATIONS
# ===============================

# 9 istasyon
L = list(range(1, 10))  # [1,...,9]

# Şimdilik tüm operasyonlar tüm istasyonlara gidebiliyor
L_i = {i: L[:] for i in I}

# Big and small stations
# 1, 2, 3 büyük istasyon; 4–9 küçük istasyon
L_big   = [1, 2, 3]
L_small = [4, 5, 6, 7, 8, 9]

# ===============================
#  PRECEDENCE – dallı/birleşmeli yapı
# ===============================

# ===============================
#  PRECEDENCE – daha heterojen yapı
# ===============================

Pred_i = {i: [] for i in I}

# 1) Temel: her job içinde chain
#    i_k-1 → i_k
for j in J:
    ops = O_j[j]
    for idx in range(1, len(ops)):
        prev_op = ops[idx - 1]
        cur_op  = ops[idx]
        Pred_i[cur_op].append(prev_op)

# 2) Ek oklar (fork / merge) – job bazlı heterojen yapı

# ---- Job 1: ops 1–18 (biraz dallanma + skip oklar)
extra_arcs_job1 = [
    (1, 3), (1, 4),
    (2, 5),
    (4, 7),
    (5, 8),
    (6, 9),
    (8, 11),
    (10, 13),
    (12, 15),
    (14, 17),
]
for h, i_ in extra_arcs_job1:
    Pred_i[i_].append(h)

# ---- Job 2: ops 19–40 (daha uzun ama başka pattern)
extra_arcs_job2 = [
    (19, 21), (19, 22),
    (20, 23),
    (22, 25),
    (24, 27),
    (26, 29),
    (28, 31),
    (30, 33),
    (32, 35),
    (34, 37),
    (36, 39),
]
for h, i_ in extra_arcs_job2:
    Pred_i[i_].append(h)

# ---- Job 3: ops 41–59 (biraz farklı dağılım)
extra_arcs_job3 = [
    (41, 43), (41, 44),
    (42, 45),
    (44, 47),
    (46, 49),
    (48, 51),
    (50, 53),
    (52, 55),
    (54, 57),
    (55, 58),
]
for h, i_ in extra_arcs_job3:
    Pred_i[i_].append(h)

# ---- Job 4: ops 60–80 (son job en büyük, başka pattern)
extra_arcs_job4 = [
    (60, 62), (60, 63),
    (61, 64),
    (63, 66),
    (65, 68),
    (67, 70),
    (69, 72),
    (71, 74),
    (73, 76),
    (75, 78),
    (77, 80),
]
for h, i_ in extra_arcs_job4:
    Pred_i[i_].append(h)

# 3) EK ADIM:
# Her job'un SON operasyonu, o jobtaki TÜM operasyonlardan sonra gelsin
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
#  PROCESSING TIMES p_im
# ===============================

p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 4)               # 3–6
        machine_factor = 0.15 * (m - 1)  # makine indexine göre küçük fark
        p_im[(i, m)] = float(base + machine_factor)

# ===============================
#  RELEASE TIMES – staggered by job
# ===============================

r_j = {
    1: 0.0,
    2: 5.0,
    3: 10.0,
    4: 15.0,
}

# ===============================
#  DUE DATES – increasing by job
# ===============================

Iend = [O_j[j][-1] for j in J]   # [20, 40, 60, 80]

d_j = {
    1: 150.0,
    2: 170.0,
    3: 190.0,
    4: 210.0,
}

# ===============================
#  GRINDING REQUIREMENTS
# ===============================

g_j = {
    1: 1,
    2: 0,
    3: 1,
    4: 0,
}

# ===============================
#  PAINTING REQUIREMENTS
# ===============================

p_j = {
    1: 1,
    2: 1,
    3: 0,
    4: 0,
}

# ===============================
#  GRINDING & PAINTING TIMES
# ===============================

t_grind_j = {}
t_paint_j = {}

for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)       # 2 or 3
    t_paint_j[j] = 3.0 if p_j[j] == 1 else 0.0

# ===============================
#  BIG-STATION REQUIREMENT
# only last ops must be on big stations
# ===============================

beta_i = {i: (1 if i in Iend else 0) for i in I}

# ===============================
# Big-M values
# ===============================

M_proc = 1000.0
M_seq  = 1000.0
M_Lseq = 1000.0


# ==========================================================
#  DATA VERIFICATION + VISUALIZATION FUNCTIONS
# ==========================================================

def verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j):
    print("\n=== DATA VERIFICATION ===")

    # 1) O_j içindeki tüm operasyonlar I'de mi?
    all_ops_from_jobs = set()
    for j in J:
        for i in O_j[j]:
            all_ops_from_jobs.add(i)
            if i not in I:
                print(f"[WARNING] Job {j} operation {i} not in I!")
    missing_ops = set(I) - all_ops_from_jobs
    if missing_ops:
        print(f"[WARNING] Some operations in I are not in any job O_j: {missing_ops}")
    else:
        print("OK: All operations in I are covered by some O_j.")

    # 2) Her op için feasible machine & station var mı?
    bad_m = [i for i in I if len(M_i[i]) == 0]
    bad_l = [i for i in I if len(L_i[i]) == 0]
    if bad_m:
        print(f"[ERROR] Some operations have no feasible machines: {bad_m}")
    else:
        print("OK: Every operation has at least one feasible machine.")
    if bad_l:
        print(f"[ERROR] Some operations have no feasible stations: {bad_l}")
    else:
        print("OK: Every operation has at least one feasible station.")

    # 3) Pred_i referansları valid mi?
    bad_pred = []
    for i in I:
        for h in Pred_i[i]:
            if h not in I:
                bad_pred.append((i, h))
    if bad_pred:
        print(f"[ERROR] Some Pred_i refer to non-existing operations: {bad_pred}")
    else:
        print("OK: All predecessors in Pred_i are valid operations.")

    # 4) p_im tanımlı mı?
    missing_p = []
    for i in I:
        for m in M_i[i]:
            if (i, m) not in p_im:
                missing_p.append((i, m))
    if missing_p:
        print(f"[ERROR] Missing p_im entries for: {missing_p}")
    else:
        print("OK: All (i,m) with m in M_i[i] have p_im defined.")

    # 5) r_j ve d_j tam mı?
    if set(r_j.keys()) != set(J):
        print("[ERROR] r_j is not defined for all jobs!")
    else:
        print("OK: r_j defined for all jobs.")
    if set(d_j.keys()) != set(J):
        print("[ERROR] d_j is not defined for all jobs!")
    else:
        print("OK: d_j defined for all jobs.")


def visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_j, t_grind_j, t_paint_j):
    """
    Çizilenler:
    - Number of operations per job
    - Release and due dates per job
    """
    print("\n=== JOB – OP SUMMARY ===")
    for j in J:
        ops = O_j[j]
        print(f"Job {j}: ops = {ops}, "
              f"r_j = {r_j[j]}, d_j = {d_j[j]}, "
              f"grind = {g_j[j]}, paint = {p_j[j]}, "
              f"t_grind = {t_grind_j[j]}, t_paint = {t_paint_j[j]}")
    
    # 1) Number of operations per job
    job_ids = J
    op_counts = [len(O_j[j]) for j in J]
    
    plt.figure()
    plt.bar(job_ids, op_counts)
    plt.xlabel("Job")
    plt.ylabel("Number of operations")
    plt.title("Number of operations per job")
    plt.tight_layout()
    
    # 2) Release and due dates per job
    releases = [r_j[j] for j in J]
    dues     = [d_j[j] for j in J]
    
    plt.figure()
    plt.scatter(job_ids, releases, marker="o", label="release")
    plt.scatter(job_ids, dues,     marker="x", label="due")
    plt.xlabel("Job")
    plt.ylabel("Time")
    plt.title("Release and due dates per job")
    plt.legend()
    plt.tight_layout()


def print_precedence_by_job(J, O_j, Pred_i):
    """
    Her job için: her operasyonun predecessor listesini yazdır.
    """
    print("\n=== PRECEDENCE BY JOB ===")
    for j in J:
        print(f"\nJob {j}: ops = {O_j[j]}")
        for i in O_j[j]:
            preds = Pred_i[i]
            print(f"  Op {i}: preds = {preds}")


def visualize_precedence_per_job(J, O_j, Pred_i):
    """
    Her job için daha okunaklı precedence grafiği:
    - X ekseni: 'seviye' (level) – daha erken operasyonlar solda, daha geç olanlar sağda
    - Y ekseni: aynı seviyedeki operasyonlar dikeyde ayrılmış
    - Oklar: h -> i bağları
    - Yeşil: job içindeki ilk operasyon
    - Kırmızı: job içindeki son operasyon
    """
    import matplotlib.lines as mlines

    for j in J:
        ops = O_j[j]

        # Sadece bu job içindeki öncelikleri al
        preds_in_job = {
            i: [h for h in Pred_i[i] if h in ops]
            for i in ops
        }

        # --- 1) Level (katman) hesapla (topolojik) ---
        level = {}
        remaining = set(ops)

        # Basit bir topolojik seviye hesaplama:
        # pred'i olmayanlar 0, diğerleri max(pred level) + 1
        while remaining:
            progress = False
            for i in list(remaining):
                preds = preds_in_job[i]
                if not preds:
                    # hiç predecessor yoksa level 0
                    level[i] = 0
                    remaining.remove(i)
                    progress = True
                elif all(p in level for p in preds):
                    # tüm predecessor'ların seviyesi belli ise
                    level[i] = max(level[p] for p in preds) + 1
                    remaining.remove(i)
                    progress = True

            # Döngüde ilerleme olmadıysa (örneğin cycle varsa)
            if not progress:
                # Kalanlara 0 ver, kırıl (bizim datada cycle yok zaten)
                for i in remaining:
                    level[i] = 0
                break

        max_level = max(level.values()) if level else 0

        # --- 2) Level -> node listesi ---
        nodes_by_level = {}
        for i, lvl in level.items():
            nodes_by_level.setdefault(lvl, []).append(i)

        # --- 3) Koordinat ata (x = level, y = o level içindeki index) ---
        pos = {}
        for lvl in range(max_level + 1):
            layer = sorted(nodes_by_level.get(lvl, []))
            k = len(layer)
            if k == 0:
                continue
            # ortalayarak yerleştir
            for idx, i in enumerate(layer):
                y = idx - (k - 1) / 2.0
                pos[i] = (lvl, y)

        # --- 4) Çizim ---
        fig_width = max(4, 1.5 * (max_level + 1))
        fig_height = max(3, 0.8 * len(ops))
        plt.figure(figsize=(fig_width, fig_height))

        first_op = ops[0]
        last_op = ops[-1]

        # Düğümler
        for i in ops:
            x, y = pos[i]
            if i == first_op:
                color = "tab:green"
            elif i == last_op:
                color = "tab:red"
            else:
                color = "tab:blue"

            plt.scatter(x, y, s=300, color=color, zorder=3, edgecolors="black")
            plt.text(x, y, str(i), ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")

        # Oklar (h -> i)
        for i in ops:
            for h in preds_in_job[i]:
                x1, y1 = pos[h]
                x2, y2 = pos[i]
                if x1 == x2:
                    x2 += 0.2
                plt.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        linewidth=1.5,
                    ),
                    zorder=2
                )

        plt.title(f"Job {j} – Precedence graph (layered)")
        plt.axis("off")

        l_first = mlines.Line2D([], [], color="tab:green", marker='o', linestyle='None',
                                markersize=8, label="First op")
        l_last = mlines.Line2D([], [], color="tab:red", marker='o', linestyle='None',
                               markersize=8, label="Last op")
        l_other = mlines.Line2D([], [], color="tab:blue", marker='o', linestyle='None',
                                markersize=8, label="Other ops")
        plt.legend(handles=[l_first, l_last, l_other], loc="upper left")

        plt.tight_layout()


def visualize_precedence_matrix_with_job_blocks(I, Pred_i, O_j, J):
    """
    GLOBAL precedence matrix:
    - matrix[h, i] = 1 if h -> i
    - job blok sınırlarını çiz
    """
    print("\n=== GLOBAL PRECEDENCE LIST (h -> i) ===")
    for i in I:
        for h in Pred_i[i]:
            print(f"{h} -> {i}")

    n = len(I)
    index_of = {op: idx for idx, op in enumerate(I)}
    mat = [[0] * n for _ in range(n)]
    for i in I:
        ci = index_of[i]
        for h in Pred_i[i]:
            if h in index_of:
                rh = index_of[h]
                mat[rh][ci] = 1

    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Precedence (1 = h -> i)")
    plt.xticks(range(n), I, rotation=90)
    plt.yticks(range(n), I)
    plt.xlabel("Successor operation i")
    plt.ylabel("Predecessor operation h")
    plt.title("Precedence matrix with job blocks")

    pos = 0
    for j in J:
        size = len(O_j[j])
        plt.axvline(pos - 0.5, linestyle="--")
        plt.axhline(pos - 0.5, linestyle="--")
        pos += size
    plt.axvline(pos - 0.5, linestyle="--")
    plt.axhline(pos - 0.5, linestyle="--")

    plt.tight_layout()


def plot_gantt_by_machine(I, M, M_i, x, S, C, title="Machine-wise welding schedule (CBC)"):
    """
    Her makine için zaman ekseninde operasyon bloklarını çiz.
    Pyomo değişkenleri üzerinden değer okur.
    """
    plt.figure()
    y_ticks = []
    y_labels = []

    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        y_labels.append(f"Machine {m}")

        for i in I:
            if m not in M_i[i]:
                continue
            if pyo.value(x[i, m]) > 0.5:
                start = pyo.value(S[i])
                finish = pyo.value(C[i])
                plt.barh(y_pos, finish - start, left=start)
                plt.text(start, y_pos, f"{i}", va="center")

    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


def plot_gantt_by_station(I, L, L_i, y, S, C, title="Station-wise welding schedule (CBC)"):
    """
    Her istasyon için zaman ekseninde operasyon bloklarını çiz.
    Pyomo değişkenleri üzerinden değer okur.
    """
    plt.figure()
    y_ticks = []
    y_labels = []

    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)
        y_labels.append(f"Station {l}")

        for i in I:
            if l not in L_i[i]:
                continue
            if pyo.value(y[i, l]) > 0.5:
                start = pyo.value(S[i])
                finish = pyo.value(C[i])
                plt.barh(y_pos, finish - start, left=start)
                plt.text(start, y_pos, f"{i}", va="center")

    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


# ===============================
#  CBC RESULT SUMMARY (GAP vs Gurobi-style)
# ===============================

def summarize_cbc_results(results, model, obj_expr, tol=1e-8):
    """
    CBC çözüm özetini Gurobi tarzı yazdırır:
    - Solver status
    - Termination condition
    - Objective value
    - Best bound (varsa)
    - GAP (varsa)
    - Runtime (varsa)
    - Node sayısı (bilgi varsa)
    """
    print("\n========== SOLVER SUMMARY (CBC) ==========")

    # Status & termination
    try:
        status = results.solver.status
    except Exception:
        status = "UNKNOWN"

    try:
        term_cond = results.solver.termination_condition
    except Exception:
        term_cond = "UNKNOWN"

    print(f"Status              : {status}")
    print(f"Termination cond.   : {term_cond}")

    # Objective değeri
    try:
        obj_value = pyo.value(obj_expr)
        print(f"Objective value     : {obj_value:.6f}")
    except Exception as e:
        print("Objective value     : (hesaplanamadı)", e)
        obj_value = None

    # Best bound (solver raporladıysa)
    best_bound = None
    try:
        problems = results.problem if isinstance(results.problem, list) else [results.problem]
        for prob in problems:
            if hasattr(prob, "lower_bound") and prob.lower_bound is not None:
                best_bound = prob.lower_bound
            elif hasattr(prob, "upper_bound") and prob.upper_bound is not None:
                best_bound = prob.upper_bound
    except Exception:
        pass

    if best_bound is not None:
        print(f"Best bound          : {best_bound:.6f}")
    else:
        print("Best bound          : (solver raporlamadı)")

    # GAP
    if (best_bound is not None) and (obj_value is not None):
        gap = abs(obj_value - best_bound) / (abs(obj_value) + tol)
        print(f"MIP GAP             : {100.0 * gap:.2f} %")
    else:
        print("MIP GAP             : (hesaplanamadı – bound veya objective yok)")

    # Runtime
    try:
        runtime = results.solver.time
        print(f"Runtime             : {runtime:.3f} s")
    except Exception:
        print("Runtime             : (solver time bilgisi yok)")

    # Node sayısı (varsa)
    node_info = None
    try:
        stats = results.solver.statistics
        if hasattr(stats, "branch_and_bound") and \
           hasattr(stats.branch_and_bound, "number_of_created_subproblems"):
            node_info = stats.branch_and_bound.number_of_created_subproblems
    except Exception:
        pass

    if node_info is not None:
        print(f"Branch&Bound nodes  : {int(node_info)}")
    else:
        print("Branch&Bound nodes  : (bilgi yok)")

    print("==========================================\n")


# ===============================
#  PYOMO MODEL
# ===============================

model = pyo.ConcreteModel()

# Set'ler
model.I = pyo.Set(initialize=I)
model.J = pyo.Set(initialize=J)
model.M = pyo.Set(initialize=M)
model.L = pyo.Set(initialize=L)

# Feasible (i,m) ve (i,l) çiftleri
MI_pairs = [(i, m) for i in I for m in M_i[i]]
LI_pairs = [(i, l) for i in I for l in L_i[i]]
model.MI = pyo.Set(dimen=2, initialize=MI_pairs)
model.LI = pyo.Set(dimen=2, initialize=LI_pairs)

# Karar değişkenleri
model.x = pyo.Var(model.MI, domain=pyo.Binary)              # x_{i,m}
model.y = pyo.Var(model.LI, domain=pyo.Binary)              # y_{i,l}

model.zM = pyo.Var(model.I, model.I, model.M, domain=pyo.Binary)  # z_{i,h,m}
model.zL = pyo.Var(model.I, model.I, model.L, domain=pyo.Binary)  # z_{i,h,l}

model.S = pyo.Var(model.I, domain=pyo.NonNegativeReals)
model.C = pyo.Var(model.I, domain=pyo.NonNegativeReals)

model.C_weld  = pyo.Var(model.J, domain=pyo.NonNegativeReals)
model.C_final = pyo.Var(model.J, domain=pyo.NonNegativeReals)

model.T     = pyo.Var(model.J, domain=pyo.NonNegativeReals)
model.T_max = pyo.Var(domain=pyo.NonNegativeReals)
model.C_max = pyo.Var(domain=pyo.NonNegativeReals)

# OBJECTIVE: min T_max
def obj_rule(m):
    return m.T_max
model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# ===============================
#  CONSTRAINTS
# ===============================

# 1) Machine assignment
model.con_assign_M = pyo.ConstraintList()
for i in I:
    model.con_assign_M.add(
        sum(model.x[i, m] for m in M_i[i]) == 1
    )

# 2) Station assignment
model.con_assign_L = pyo.ConstraintList()
for i in I:
    model.con_assign_L.add(
        sum(model.y[i, l] for l in L_i[i]) == 1
    )

# 3) Processing time linking
model.con_proc = pyo.ConstraintList()
for i in I:
    for m in M_i[i]:
        pijm = p_im[(i, m)]
        model.con_proc.add(
            model.C[i] - model.S[i] >= pijm - M_proc * (1 - model.x[i, m])
        )
        model.con_proc.add(
            model.C[i] - model.S[i] <= pijm + M_proc * (1 - model.x[i, m])
        )

# 4) Station size feasibility (beta_i == 1 ise küçük istasyona gidemez)
model.con_station_size = pyo.ConstraintList()
for i in I:
    if beta_i[i] == 1:
        for l in L_small:
            if l in L_i[i]:
                model.con_station_size.add(model.y[i, l] == 0)

# 5) Precedence
model.con_prec = pyo.ConstraintList()
for i in I:
    for h in Pred_i[i]:
        model.con_prec.add(model.S[i] >= model.C[h])

# 6) Earliest start
model.con_release = pyo.ConstraintList()
for j in J:
    first_op = O_j[j][0]
    model.con_release.add(model.S[first_op] >= r_j[j])

# 7–8) Machine sequencing
model.con_seqM = pyo.ConstraintList()
I_list = I[:]
nI = len(I_list)

for idx_i in range(nI):
    for idx_h in range(idx_i + 1, nI):
        i = I_list[idx_i]
        h = I_list[idx_h]

        common_machines = set(M_i[i]).intersection(M_i[h])

        for m in common_machines:
            # h after i
            model.con_seqM.add(
                model.S[h] >= model.C[i]
                             - M_seq * (1 - model.zM[i, h, m])
                             - M_seq * (1 - model.x[i, m])
                             - M_seq * (1 - model.x[h, m])
            )
            # i after h
            model.con_seqM.add(
                model.S[i] >= model.C[h]
                             - M_seq * model.zM[i, h, m]
                             - M_seq * (1 - model.x[i, m])
                             - M_seq * (1 - model.x[h, m])
            )

            model.con_seqM.add(model.zM[i, h, m] <= model.x[i, m])
            model.con_seqM.add(model.zM[i, h, m] <= model.x[h, m])
            model.con_seqM.add(
                model.zM[i, h, m] >= model.x[i, m] + model.x[h, m] - 1
            )

# 9–10) Station sequencing
model.con_seqL = pyo.ConstraintList()
for idx_i in range(nI):
    for idx_h in range(idx_i + 1, nI):
        i = I_list[idx_i]
        h = I_list[idx_h]

        common_stations = set(L_i[i]).intersection(L_i[h])

        for l in common_stations:
            # h after i
            model.con_seqL.add(
                model.S[h] >= model.C[i]
                             - M_Lseq * (1 - model.zL[i, h, l])
                             - M_Lseq * (1 - model.y[i, l])
                             - M_Lseq * (1 - model.y[h, l])
            )
            # i after h
            model.con_seqL.add(
                model.S[i] >= model.C[h]
                             - M_Lseq * model.zL[i, h, l]
                             - M_Lseq * (1 - model.y[i, l])
                             - M_Lseq * (1 - model.y[h, l])
            )

            model.con_seqL.add(model.zL[i, h, l] <= model.y[i, l])
            model.con_seqL.add(model.zL[i, h, l] <= model.y[h, l])
            model.con_seqL.add(
                model.zL[i, h, l] >= model.y[i, l] + model.y[h, l] - 1
            )

# 11) Welding completion
model.con_Cweld = pyo.ConstraintList()
for j in J:
    last_op = O_j[j][-1]
    model.con_Cweld.add(model.C_weld[j] == model.C[last_op])

# 12) Final completion
model.con_Cfinal = pyo.ConstraintList()
for j in J:
    model.con_Cfinal.add(
        model.C_final[j] ==
        model.C_weld[j]
        + g_j[j] * t_grind_j[j]
        + p_j[j] * t_paint_j[j]
    )

# 13) Tardiness
model.con_Tdef = pyo.ConstraintList()
for j in J:
    model.con_Tdef.add(model.T[j] >= model.C_final[j] - d_j[j])

# 14) Makespan and maximum tardiness
model.con_Tmax = pyo.ConstraintList()
for j in J:
    model.con_Tmax.add(model.C_max >= model.C_final[j])
    model.con_Tmax.add(model.T_max >= model.T[j])

# ===============================
#  DATA VERIFICATION + VISUALIZATION (before solve)
# ===============================

verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j)
visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_j, t_grind_j, t_paint_j)

print_precedence_by_job(J, O_j, Pred_i)
visualize_precedence_per_job(J, O_j, Pred_i)
visualize_precedence_matrix_with_job_blocks(I, Pred_i, O_j, J)

# ===============================
#  SOLVE (CBC)
# ===============================

solver = pyo.SolverFactory(
    "cbc",
    executable=r"C:\cbc\bin\cbc.exe"
)
results = solver.solve(model, tee=True)

# CBC çözüm özeti (GAP vs bound vs süre)
summarize_cbc_results(results, model, model.OBJ)

# ===============================
#  PRINT, CHECK & GANTT
# ===============================

tc = results.solver.termination_condition

if tc not in [TerminationCondition.optimal, TerminationCondition.feasible]:
    print("\nNo feasible/optimal solution. Termination condition:", tc)

else:
    check_solution(
        J, I, O_j,
        M, L,
        M_i, L_i,
        Pred_i,
        r_j, d_j,
        g_j, p_j,
        t_grind_j, t_paint_j,
        model.x, model.y, model.S, model.C,
        model.C_weld, model.C_final,
        model.T, model.T_max, model.C_max,
        tol=1e-6
    )
    print("\n===== Objective =====")
    print(f"T_max = {pyo.value(model.T_max):.2f}")
    print(f"C_max = {pyo.value(model.C_max):.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld = {pyo.value(model.C_weld[j]):.2f}, "
              f"C_final = {pyo.value(model.C_final[j]):.2f}, "
              f"T_j = {pyo.value(model.T[j]):.2f}, d_j = {d_j[j]}")

    print("\n===== Operations =====")
    for i in I:
        m_list = [m for m in M_i[i] if pyo.value(model.x[i, m]) > 0.5]
        l_list = [l for l in L_i[i] if pyo.value(model.y[i, l]) > 0.5]
        m_sel = m_list[0] if m_list else None
        l_sel = l_list[0] if l_list else None
        print(f"Op {i}: S = {pyo.value(model.S[i]):.2f}, "
              f"C = {pyo.value(model.C[i]):.2f}, "
              f"machine = {m_sel}, station = {l_sel}")

    plot_gantt_by_machine(I, M, M_i, model.x, model.S, model.C,
                          title="Machine-wise welding schedule (CBC)")
    plot_gantt_by_station(I, L, L_i, model.y, model.S, model.C,
                          title="Station-wise welding schedule (CBC)")

    plt.show()
