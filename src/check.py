import pyomo.environ as pyo

def check_solution(
    J, I, O_j, M, L, M_i, L_i, Pred_i,
    r_j, d_j, g_j, p_j, t_grind_j, t_paint_j,
    x, y, S, C, C_weld, C_final, T, T_max, C_max,
    tol=1e-6
):
    """
    Çözüm feasibility check (Pyomo versiyonu):
    - Makine/istasyon atamaları
    - Precedence
    - Release/due/tardiness
    - Makespan & T_max
    - Makine/istasyon çakışması (interval overlap)
    """
    print("\n================= SOLUTION CHECK =================")

    # Eğer çözüm yoksa
    try:
        _ = pyo.value(T_max)
    except Exception as e:
        print("Çözüm okunamadı (muhtemelen model çözülmedi veya infeasible). Hata:", e)
        return

    # -------------------------
    # 1) Makine / İstasyon atamaları
    # -------------------------
    viol_assign_M = []
    viol_assign_L = []

    for i in I:
        sumM = sum(pyo.value(x[i, m]) for m in M_i[i])
        sumL = sum(pyo.value(y[i, l]) for l in L_i[i])
        if abs(sumM - 1.0) > tol:
            viol_assign_M.append((i, sumM))
        if abs(sumL - 1.0) > tol:
            viol_assign_L.append((i, sumL))

    if not viol_assign_M:
        print("OK: Her operasyon tam olarak 1 makineye atanmış.")
    else:
        print("!! HATA (Makine ataması): Aşağıdaki operasyonlarda sum_m x[i,m] != 1:")
        for i, val in viol_assign_M[:10]:
            print(f"   Op {i}: sum_m x[i,m] = {val}")
        if len(viol_assign_M) > 10:
            print(f"   ... ve {len(viol_assign_M) - 10} tane daha")

    if not viol_assign_L:
        print("OK: Her operasyon tam olarak 1 istasyona atanmış.")
    else:
        print("!! HATA (İstasyon ataması): Aşağıdaki operasyonlarda sum_l y[i,l] != 1:")
        for i, val in viol_assign_L[:10]:
            print(f"   Op {i}: sum_l y[i,l] = {val}")
        if len(viol_assign_L) > 10:
            print(f"   ... ve {len(viol_assign_L) - 10} tane daha")

    # -------------------------
    # 2) Precedence
    # -------------------------
    viol_prec = []
    for i in I:
        for h in Pred_i[i]:
            if pyo.value(S[i]) + tol < pyo.value(C[h]):
                viol_prec.append((h, i, pyo.value(C[h]), pyo.value(S[i])))

    if not viol_prec:
        print("OK: Tüm precedence kısıtları (S[i] >= C[h]) sağlanmış.")
    else:
        print("!! HATA (Precedence): Aşağıdaki (h -> i) çiftlerinde ihlal var (C[h] > S[i]):")
        for h, i, Ch, Si in viol_prec[:10]:
            print(f"   {h} -> {i}: C[{h}]={Ch:.3f}, S[{i}]={Si:.3f}")
        if len(viol_prec) > 10:
            print(f"   ... ve {len(viol_prec) - 10} tane daha")

    # -------------------------
    # 3) Release, C_weld, C_final, T, T_max, C_max
    # -------------------------
    viol_release = []
    viol_Cweld   = []
    viol_Cfinal  = []
    viol_Tdef    = []
    viol_Tmax    = []
    viol_Cmax    = []

    # Release & C_weld & C_final & T
    for j in J:
        first_op = O_j[j][0]
        last_op  = O_j[j][-1]

        # release: S[first_op] >= r_j[j]
        if pyo.value(S[first_op]) + tol < r_j[j]:
            viol_release.append((j, first_op, pyo.value(S[first_op]), r_j[j]))

        # C_weld[j] == C[last_op]
        if abs(pyo.value(C_weld[j]) - pyo.value(C[last_op])) > tol:
            viol_Cweld.append((j, pyo.value(C_weld[j]), pyo.value(C[last_op])))

        # C_final[j] == C_weld[j] + g_j[j]*t_grind_j[j] + p_j[j]*t_paint_j[j]
        rhs = pyo.value(C_weld[j]) + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j]
        if abs(pyo.value(C_final[j]) - rhs) > tol:
            viol_Cfinal.append((j, pyo.value(C_final[j]), rhs))

        # T[j] >= C_final[j] - d_j[j]
        if pyo.value(T[j]) + tol < pyo.value(C_final[j]) - d_j[j]:
            viol_Tdef.append((j, pyo.value(T[j]), pyo.value(C_final[j]), d_j[j]))

    # T_max, C_max
    max_T = max(pyo.value(T[j]) for j in J)
    max_Cf = max(pyo.value(C_final[j]) for j in J)

    if pyo.value(T_max) + tol < max_T:
        viol_Tmax.append((pyo.value(T_max), max_T))
    if pyo.value(C_max) + tol < max_Cf:
        viol_Cmax.append((pyo.value(C_max), max_Cf))

    if not viol_release:
        print("OK: Release zamanları (S[first_op] >= r_j) sağlanmış.")
    else:
        print("!! HATA (Release):")
        for j, i, Si, rj in viol_release:
            print(f"   Job {j}, first op {i}: S={Si:.3f}, r_j={rj:.3f}")

    if not viol_Cweld:
        print("OK: C_weld[j] = son operasyonun C[i] değeri.")
    else:
        print("!! HATA (C_weld tanımı):")
        for j, Cwj, Clast in viol_Cweld:
            print(f"   Job {j}: C_weld={Cwj:.3f}, C[last_op]={Clast:.3f}")

    if not viol_Cfinal:
        print("OK: C_final[j] = C_weld[j] + grinding + painting.")
    else:
        print("!! HATA (C_final tanımı):")
        for j, Cf, rhs in viol_Cfinal:
            print(f"   Job {j}: C_final={Cf:.3f}, RHS={rhs:.3f}")

    if not viol_Tdef:
        print("OK: T[j] >= C_final[j] - d_j[j] kısıtları sağlanmış.")
    else:
        print("!! HATA (Tardiness tanımı):")
        for j, Tj, Cf, dj in viol_Tdef:
            print(f"   Job {j}: T={Tj:.3f}, C_final={Cf:.3f}, d_j={dj:.3f}")

    if not viol_Tmax:
        print("OK: T_max >= T[j] (tüm j) sağlanmış.")
    else:
        print("!! HATA (T_max):")
        Tmx, maxT = viol_Tmax[0]
        print(f"   T_max={Tmx:.3f}, ama max_j T[j]={maxT:.3f}")

    if not viol_Cmax:
        print("OK: C_max >= C_final[j] (tüm j) sağlanmış.")
    else:
        print("!! HATA (C_max):")
        Cmx, maxCf = viol_Cmax[0]
        print(f"   C_max={Cmx:.3f}, ama max_j C_final[j]={maxCf:.3f}")

    # -------------------------
    # 4) Makine çakışması (interval overlap)
    # -------------------------
    viol_machine_overlap = []

    for m in M:
        # Bu makinaya atanmış operasyonları topla
        ops_m = []
        for i in I:
            if m in M_i[i] and pyo.value(x[i, m]) > 0.5:
                ops_m.append((pyo.value(S[i]), pyo.value(C[i]), i))
        # Başlangıca göre sırala
        ops_m.sort(key=lambda t: t[0])

        for k in range(len(ops_m) - 1):
            s1, c1, i1 = ops_m[k]
            s2, c2, i2 = ops_m[k+1]
            if s2 + tol < c1:  # ikinci, birinci bitmeden başlarsa overlap
                viol_machine_overlap.append((m, i1, i2, s1, c1, s2, c2))

    if not viol_machine_overlap:
        print("OK: Makine bazında zaman çakışması yok.")
    else:
        print("!! HATA (Makine overlap): Aynı makinede çakışan operasyonlar var.")
        for m, i1, i2, s1, c1, s2, c2 in viol_machine_overlap[:10]:
            print(f"   M{m}: Op {i1} [{s1:.3f}, {c1:.3f}] ile "
                  f"Op {i2} [{s2:.3f}, {c2:.3f}] çakışıyor.")
        if len(viol_machine_overlap) > 10:
            print(f"   ... ve {len(viol_machine_overlap) - 10} tane daha")

    # -------------------------
    # 5) İstasyon çakışması (interval overlap)
    # -------------------------
    viol_station_overlap = []

    for l in L:
        # Bu istasyona atanmış operasyonları topla
        ops_l = []
        for i in I:
            if l in L_i[i] and pyo.value(y[i, l]) > 0.5:
                ops_l.append((pyo.value(S[i]), pyo.value(C[i]), i))
        ops_l.sort(key=lambda t: t[0])

        for k in range(len(ops_l) - 1):
            s1, c1, i1 = ops_l[k]
            s2, c2, i2 = ops_l[k+1]
            if s2 + tol < c1:
                viol_station_overlap.append((l, i1, i2, s1, c1, s2, c2))

    if not viol_station_overlap:
        print("OK: İstasyon bazında zaman çakışması yok.")
    else:
        print("!! HATA (İstasyon overlap): Aynı istasyonda çakışan operasyonlar var.")
        for l, i1, i2, s1, c1, s2, c2 in viol_station_overlap[:10]:
            print(f"   L{l}: Op {i1} [{s1:.3f}, {c1:.3f}] ile "
                  f"Op {i2} [{s2:.3f}, {c2:.3f}] çakışıyor.")
        if len(viol_station_overlap) > 10:
            print(f"   ... ve {len(viol_station_overlap) - 10} tane daha")

    print("================= END SOLUTION CHECK =================\n")
