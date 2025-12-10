# -- coding: utf-8 --
"""
Welding Shop Scheduling – LARGE EXAMPLE (8 jobs, 30 ops)
Verification + visualization:

- Number of operations per job
- Release and due dates per job
- GLOBAL precedence table + matrix (all operations)
- Machine-wise welding schedule
- Station-wise welding schedule

SCENARIO: Most operations require BIG stations
"""

#!/usr/bin/env python3
# -- coding: utf-8 --

from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from check_gurobi import check_solution_gurobi

# ===============================
#  DATA (LARGE EXAMPLE – 8 jobs, 30 ops)
# ===============================

# Jobs
J = [1, 2, 3, 4, 5, 6, 7, 8]

# Operations
I = list(range(1, 31))  # 1..30

# Job -> operations mapping (O_j)
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
#  MACHINES (12 pcs, 2 TYPES) & STATIONS
# ===============================

# 12 welding machines
M = list(range(1, 13))  # 1..12

# 2 machine types:
# 1 = TIG welding
# 2 = MAG welding
K = [1, 2]

# Each machine's type
# Machines 1-3: TIG, Machines 4-12: MAG
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
# (şimdilik örnek pattern: tekler TIG, çiftler MAG)
K_i = {}
for i in I:
    if i % 2 == 1:
        K_i[i] = [1]      # TIG
    else:
        K_i[i] = [2]      # MAG

# Feasible machine sets M_i are induced by types:
# M_i[i] = { m in M : machine_type[m] in K_i[i] }
M_i = {
    i: [m for m in M if machine_type[m] in K_i[i]]
    for i in I
}

# ==== STATIONS ====

L = [1, 2, 3, 4]

# Feasible station sets L_i (her operasyon her istasyona gidebilir)
L_i = {i: [1, 2, 3, 4] for i in I}

# Big and small stations
# Only Station 1 is big, 2–4 are small
L_big   = [1]
L_small = [2, 3, 4]

# ===============================
#  PRECEDENCE (bazıları bağlı, bazıları bağımsız)
# ===============================

Pred_i = {i: [] for i in I}

# Job 1: ops = [1, 2, 3]
# 1 → 2, 1 → 3
Pred_i[2] = [1]
Pred_i[3] = [1]

# Job 2: ops = [4, 5, 6, 7]
# 4 → 5, 4 → 6, 5 → 7
Pred_i[5] = [4]
Pred_i[6] = [4]
Pred_i[7] = [5]

# Job 3: ops = [8, 9, 10, 11, 12]
# 8 → 9, 8 → 10, 9 & 10 → 11, 11 → 12
Pred_i[9]  = [8]
Pred_i[10] = [8]
Pred_i[11] = [9, 10]
Pred_i[12] = [11]

# Job 4: ops = [13, 14]
# 13 → 14
Pred_i[14] = [13]

# Job 5: ops = [15, 16, 17, 18]
# 15 → 16, 15 → 17, 16 → 18
Pred_i[16] = [15]
Pred_i[17] = [15]
Pred_i[18] = [16]

# Job 6: ops = [19, 20, 21]
# 19 & 20 → 21
Pred_i[21] = [19, 20]

# Job 7: ops = [22, 23, 24, 25, 26]
# 22 → 23, 22 → 24, 23 → 25, 24 → 26
Pred_i[23] = [22]
Pred_i[24] = [22]
Pred_i[25] = [23]
Pred_i[26] = [24]

# Job 8: ops = [27, 28, 29, 30]
# İç bağlantı yok (şimdilik)

# EK ADIM: Her job'un son operasyonu, o jobtaki tüm operasyonlardan sonra gelsin
for j in J:
    ops = O_j[j]
    last = ops[-1]
    preds = set(Pred_i[last])
    for i_op in ops:
        if i_op == last:
            continue
        preds.add(i_op)
    Pred_i[last] = list(preds)

# Processing times p_im
p_im = {}
for i in I:
    for m in M_i[i]:
        base = 3 + (i % 5)          # 3..7 arası base
        machine_add = (m - 1) * 0.5 # makine indexine göre fark
        p_im[(i, m)] = float(base + machine_add)

# Release times r_j (0,2,4,...,14)
r_j = {}
release_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
for idx, j in enumerate(J):
    r_j[j] = release_times[idx]

# Due dates d_j
d_i = {}
due_base = 30.0
Iend = [O_j[j][-1] for j in J]  # last op of each job
for idx, i_last in enumerate(Iend):
    d_i[i_last] = due_base + 3.0 * idx   # 30,33,36,...,51

d_j = {}
for j in J:
    last_op = O_j[j][-1]
    d_j[j] = d_i[last_op]

# Grinding requirement g_j
g_j = {}
for idx, j in enumerate(J):
    if idx % 2 == 0:   # j=1,3,5,7
        g_j[j] = 1
    else:              # j=2,4,6,8
        g_j[j] = 0

# Painting requirement p_j
p_j = {}
for idx, j in enumerate(J):
    if idx in [0, 1, 2]:  # j=1,2,3
        p_j[j] = 1
    else:
        p_j[j] = 0

# Grinding and painting times
t_grind_j = {}
t_paint_j = {}
for j in J:
    last_op = O_j[j][-1]
    t_grind_j[j] = 2.0 + (last_op % 2)
    if p_j[j] == 1:
        t_paint_j[j] = 3.0
    else:
        t_paint_j[j] = 0.0

# Big-station requirement beta_i
# MOST operations require big stations.
# Only the FIRST operation of each job is allowed to use small stations.
beta_i = {}
first_ops = [O_j[j][0] for j in J]  # [1,4,8,13,15,19,22,27]
for i in I:
    if i in first_ops:
        beta_i[i] = 0   # these can use small or big
    else:
        beta_i[i] = 1   # these MUST use big (Station 1)

# Big-M values
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


def visualize_precedence_matrix(I, Pred_i):
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
    plt.title("Precedence matrix for all operations")
    plt.tight_layout()


def plot_gantt_by_machine(I, M, M_i, x, S, C, title="Machine-wise welding schedule"):
    plt.figure()
    y_ticks = []
    y_labels = []

    for idx_m, m in enumerate(M):
        y_pos = idx_m
        y_ticks.append(y_pos)
        # makine tipini TIG / MAG olarak yazalım
        mtype = "TIG" if machine_type[m] == 1 else "MAG"
        y_labels.append(f"Machine {m} ({mtype})")

        for i in I:
            if m not in M_i[i]:
                continue
            if x[i, m].X > 0.5:
                start = S[i].X
                finish = C[i].X
                plt.barh(y_pos, finish - start, left=start)
                plt.text(start, y_pos, f"{i}", va="center", fontsize=7)

    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time")
    plt.title(title)
    plt.tight_layout()


def plot_gantt_by_station(I, L, L_i, y, S, C, title="Station-wise welding schedule"):
    """
    Station-wise Gantt:
    - Solda: Gantt chart
    - Sağda: Job Color Key paneli (hangi renk hangi job)
    """
    fig, (ax, ax_leg) = plt.subplots(
        1, 2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [4, 1]}
    )

    y_ticks = []
    y_labels = []

    # Operasyon -> job map'i
    op_to_job = {}
    for j in J:
        for op in O_j[j]:
            op_to_job[op] = j

    # Job'lar için renk paleti
    cmap = plt.cm.get_cmap("tab10")
    job_colors = {j: cmap((j - 1) % 10) for j in J}

    # ---- SOL: Gantt ----
    for idx_l, l in enumerate(L):
        y_pos = idx_l
        y_ticks.append(y_pos)

        if l == 1:
            y_labels.append("Station 1 (big station)")
        else:
            y_labels.append(f"Station {l}")

        for i in I:
            if l not in L_i[i]:
                continue
            if y[i, l].X > 0.5:
                start = S[i].X
                finish = C[i].X
                width = finish - start

                job_of_i = op_to_job[i]
                color = job_colors[job_of_i]

                ax.barh(y_pos, width, left=start, color=color)
                # Bar içine sadece operasyon numarasını yaz
                ax.text(start + width / 2, y_pos, f"{i}",
                        ha="center", va="center", fontsize=7)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ---- SAĞ: Job Color Key paneli ----
    ax_leg.set_axis_off()
    ax_leg.set_title("Job Color Key", fontsize=12, pad=10)

    # Paneli aksis koordinatlarında kullanmak için
    # 0–1 arası normalize edilmiş koordinatlar
    y0 = 0.9
    dy = 0.7 / max(len(J), 1)  # boşluklara göre ölçekle

    for idx, j in enumerate(J):
        y_pos = y0 - idx * dy
        if y_pos < 0.05:
            break  # taşmasın, yeterince job gösterdik

        # Renkli kare
        rect = plt.Rectangle(
            (0.05, y_pos - 0.03),
            0.12,
            0.06,
            transform=ax_leg.transAxes,
            facecolor=job_colors[j],
            edgecolor="black"
        )
        ax_leg.add_patch(rect)

        # Yanına yazı
        ax_leg.text(
            0.22, y_pos,
            f"Job {j}",
            transform=ax_leg.transAxes,
            va="center", ha="left",
            fontsize=10
        )

    fig.tight_layout()


def visualize_big_station_needs(J, O_j, beta_i):
    """
    Big station requirement tablosu / grafiği:

    - Y ekseni: Job'lar
    - X ekseni: Job içindeki operasyon pozisyonu (1., 2., 3. ... gibi)
    - Her operasyon için ayrı bar çiziliyor (aynı job satırı üzerinde yatay).
    - Her barın içine operasyon numarası yazılıyor.
    - beta_i = 1 (big zorunlu) ve beta_i = 0 (esnek) için farklı renkler.
    """
    job_ids = []
    must_big_counts = []
    flexible_counts = []

    for j in J:
        ops = O_j[j]
        must_big = sum(1 for i in ops if beta_i[i] == 1)
        flex     = sum(1 for i in ops if beta_i[i] == 0)
        job_ids.append(j)
        must_big_counts.append(must_big)
        flexible_counts.append(flex)

    print("\n=== BIG STATION REQUIREMENT SUMMARY ===")
    for j, mb, fx in zip(job_ids, must_big_counts, flexible_counts):
        print(f"Job {j}: must_big = {mb}, flexible = {fx}, total = {mb+fx}")

    # Yatay bar grafiği: job'lar y ekseninde, operasyon "pozisyonları" x ekseninde
    plt.figure()
    ax = plt.gca()

    for j_idx, j in enumerate(J):
        ops = O_j[j]
        for pos_idx, i in enumerate(ops):
            # Her operasyon için yatay bir bar: x = pos_idx .. pos_idx+1
            left = pos_idx
            width = 0.9

            # beta_i'ye göre renk (zorunlu big vs esnek)
            if beta_i[i] == 1:
                color = "tab:orange"
            else:
                color = "tab:blue"

            ax.barh(j_idx, width, left=left, color=color)
            # Barın içine operasyon numarasını yaz
            ax.text(left + width / 2.0, j_idx, f"{i}",
                    va="center", ha="center", fontsize=8)

    ax.set_yticks(range(len(J)))
    ax.set_yticklabels(J)
    ax.set_xlabel("Operation position within job")
    ax.set_ylabel("Job")
    ax.set_title("Big-station requirement per job (operation-level)")

    # Legend (esnek vs big zorunlu)
    legend_handles = [
        Patch(facecolor="tab:blue",  label="Flexible (β_i = 0)"),
        Patch(facecolor="tab:orange", label="Must be big (β_i = 1)")
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()

# ===============================
#  MODEL
# ===============================

model = Model("Welding_Full_LaTeX_Integrated_BigStationStress")

# x_{im} only for m in M_i[i]
x = model.addVars(
    [(i, m) for i in I for m in M_i[i]],
    vtype=GRB.BINARY,
    name="x"
)

# y_{iℓ} for ℓ in L_i[i]
y = model.addVars(
    [(i, l) for i in I for l in L_i[i]],
    vtype=GRB.BINARY,
    name="y"
)

# sequencing z_{ii'm}, z_{ii'ℓ}
zM_index = [(i, h, m) for i in I for h in I if i != h for m in M]
zM = model.addVars(zM_index, vtype=GRB.BINARY, name="zM")

zL_index = [(i, h, l) for i in I for h in I if i != h for l in L]
zL = model.addVars(zL_index, vtype=GRB.BINARY, name="zL")

# times
S = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="S")
C = model.addVars(I, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

C_weld  = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="C_weld")
C_final = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="C_final")

T     = model.addVars(J, lb=0.0, vtype=GRB.CONTINUOUS, name="T")
T_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="T_max")
C_max = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")

# OBJECTIVE
model.setObjective(T_max, GRB.MINIMIZE)

# ===============================
#  CONSTRAINTS
# ===============================

# 1) Machine assignment
for i in I:
    model.addConstr(
        quicksum(x[i, m] for m in M_i[i]) == 1,
        name=f"assign_M_{i}"
    )

# 2) Station assignment
for i in I:
    model.addConstr(
        quicksum(y[i, l] for l in L_i[i]) == 1,
        name=f"assign_L_{i}"
    )

# 3) Processing time linking
for i in I:
    for m in M_i[i]:
        pijm = p_im[(i, m)]
        model.addConstr(
            C[i] - S[i] >= pijm - M_proc * (1 - x[i, m]),
            name=f"ptime_lb_{i}_{m}"
        )
        model.addConstr(
            C[i] - S[i] <= pijm + M_proc * (1 - x[i, m]),
            name=f"ptime_ub_{i}_{m}"
        )

# 4) Station size feasibility
for i in I:
    if beta_i[i] == 1:
        for l in L_small:
            if l in L_i[i]:
                model.addConstr(y[i, l] == 0, name=f"small_forbid_{i}_{l}")

# 5) Precedence
for i in I:
    for h in Pred_i[i]:
        model.addConstr(S[i] >= C[h], name=f"prec_{h}_{i}")

# 6) Earliest start
for j in J:
    first_op = O_j[j][0]
    model.addConstr(S[first_op] >= r_j[j], name=f"release_{j}")

# 7–8) Machine sequencing
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

            model.addConstr(
                zM[i, h, m] <= x[i, m],
                name=f"zM_le_xi_{i}{h}{m}"
            )
            model.addConstr(
                zM[i, h, m] <= x[h, m],
                name=f"zM_le_xh_{i}{h}{m}"
            )
            model.addConstr(
                zM[i, h, m] >= x[i, m] + x[h, m] - 1,
                name=f"zM_ge_sum_{i}{h}{m}"
            )

# 9–10) Station sequencing
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

            model.addConstr(
                zL[i, h, l] <= y[i, l],
                name=f"zL_le_yi_{i}{h}{l}"
            )
            model.addConstr(
                zL[i, h, l] <= y[h, l],
                name=f"zL_le_yh_{i}{h}{l}"
            )
            model.addConstr(
                zL[i, h, l] >= y[i, l] + y[h, l] - 1,
                name=f"zL_ge_sum_{i}{h}{l}"
            )

# 11) Welding completion
for j in J:
    last_op = O_j[j][-1]
    model.addConstr(C_weld[j] == C[last_op], name=f"Cweld_{j}")

# 12) Final completion
for j in J:
    model.addConstr(
        C_final[j] == C_weld[j]
                     + g_j[j] * t_grind_j[j]
                     + p_j[j] * t_paint_j[j],
        name=f"Cfinal_{j}"
    )

# 13) Tardiness
for j in J:
    model.addConstr(T[j] >= C_final[j] - d_j[j], name=f"Tdef_{j}")

# 14) Makespan and maximum tardiness
for j in J:
    model.addConstr(C_max >= C_final[j], name=f"Cmax_ge_{j}")
    model.addConstr(T_max >= T[j],      name=f"Tmax_ge_{j}")

# ===============================
#  DATA VERIFICATION + VISUALIZATION (before solve)
# ===============================

verify_data(J, I, O_j, M_i, L_i, Pred_i, p_im, r_j, d_j)
visualize_jobs_and_ops(J, O_j, r_j, d_j, g_j, p_j, t_grind_j, t_paint_j)
visualize_precedence_matrix(I, Pred_i)
visualize_big_station_needs(J, O_j, beta_i)

# ===============================
#  SOLVE
# ===============================

model.optimize()

# ===============================
#  PRINT & GANTT
# ===============================

if model.SolCount == 0:
    print("\nNo feasible solution. Status =", model.Status)
else:
    check_solution_gurobi(
        J, I, O_j,
        M, L,
        M_i, L_i,
        Pred_i,
        r_j, d_j,
        g_j, p_j,
        t_grind_j, t_paint_j,
        x, y, S, C,
        C_weld, C_final,
        T, T_max, C_max,
        tol=1e-6
    )
    print("\n===== Objective =====")
    print(f"T_max = {T_max.X:.2f}")
    print(f"C_max = {C_max.X:.2f}")

    print("\n===== Jobs =====")
    for j in J:
        print(f"Job {j}: C_weld = {C_weld[j].X:.2f}, "
              f"C_final = {C_final[j].X:.2f}, "
              f"T_j = {T[j].X:.2f}, d_j = {d_j[j]}")

    print("\n===== Operations =====")
    for i in I:
        m_list = [m for m in M_i[i] if x[i, m].X > 0.5]
        l_list = [l for l in L_i[i] if y[i, l].X > 0.5]
        m_sel = m_list[0] if m_list else None
        l_sel = l_list[0] if l_list else None
        if m_sel is not None:
            mtype_str = "TIG" if machine_type[m_sel] == 1 else "MAG"
        else:
            mtype_str = "-"
        print(f"Op {i}: S = {S[i].X:.2f}, C = {C[i].X:.2f}, "
              f"machine = {m_sel} ({mtype_str}), station = {l_sel}")

    # Gantt grafikleri
    plot_gantt_by_machine(I, M, M_i, x, S, C,
                          title="Machine-wise welding schedule")
    plot_gantt_by_station(I, L, L_i, y, S, C,
                          title="Station-wise welding schedule")

    plt.show()