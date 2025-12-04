# check_gurobi.py
def check_solution_gurobi(
    J, I, O_j, M, L, M_i, L_i, Pred_i,
    r_j, d_j, g_j, p_j, t_grind_j, t_paint_j,
    x, y, S, C, C_weld, C_final, T, T_max, C_max,
    tol=1e-6
):
    print("\n================= SOLUTION CHECK (GUROBI) =================")

    # çözüm var mı diye kontrol
    try:
        _ = T_max.X
    except Exception as e:
        print("Çözüm okunamadı (muhtemelen model çözülmedi veya infeasible). Hata:", e)
        return

    # 1) Makine / istasyon atamaları
    viol_assign_M = []
    viol_assign_L = []

    for i in I:
        sumM = sum(x[i, m].X for m in M_i[i])
        sumL = sum(y[i, l].X for l in L_i[i])
        if abs(sumM - 1.0) > tol:
            viol_assign_M.append((i, sumM))
        if abs(sumL - 1.0) > tol:
            viol_assign_L.append((i, sumL))

    if not viol_assign_M:
        print("OK: Her operasyon tam olarak 1 makineye atanmış.")
    else:
        print("!! HATA (Makine ataması):")
        for i, val in viol_assign_M[:10]:
            print(f"   Op {i}: sum_m x[i,m] = {val}")

    if not viol_assign_L:
        print("OK: Her operasyon tam olarak 1 istasyona atanmış.")
    else:
        print("!! HATA (İstasyon ataması):")
        for i, val in viol_assign_L[:10]:
            print(f"   Op {i}: sum_l y[i,l] = {val}")

    # 2) Precedence
    viol_prec = []
    for i in I:
        for h in Pred_i[i]:
            if S[i].X + tol < C[h].X:
                viol_prec.append((h, i, C[h].X, S[i].X))

    if not viol_prec:
        print("OK: Tüm precedence kısıtları sağlanmış.")
    else:
        print("!! HATA (Precedence):")
        for h, i, Ch, Si in viol_prec[:10]:
            print(f"   {h} -> {i}: C[{h}]={Ch:.3f}, S[{i}]={Si:.3f}")

    # 3) Release, C_weld, C_final, T, T_max, C_max
    viol_release = []
    viol_Cweld   = []
    viol_Cfinal  = []
    viol_Tdef    = []
    viol_Tmax    = []
    viol_Cmax    = []

    for j in J:
        first_op = O_j[j][0]
        last_op  = O_j[j][-1]

        if S[first_op].X + tol < r_j[j]:
            viol_release.append((j, first_op, S[first_op].X, r_j[j]))

        if abs(C_weld[j].X - C[last_op].X) > tol:
            viol_Cweld.append((j, C_weld[j].X, C[last_op].X))

        rhs = C_weld[j].X + g_j[j] * t_grind_j[j] + p_j[j] * t_paint_j[j]
        if abs(C_final[j].X - rhs) > tol:
            viol_Cfinal.append((j, C_final[j].X, rhs))

        if T[j].X + tol < C_final[j].X - d_j[j]:
            viol_Tdef.append((j, T[j].X, C_final[j].X, d_j[j]))

    max_T  = max(T[j].X for j in J)
    max_Cf = max(C_final[j].X for j in J)

    if T_max.X + tol < max_T:
        viol_Tmax.append((T_max.X, max_T))
    if C_max.X + tol < max_Cf:
        viol_Cmax.append((C_max.X, max_Cf))

    # (buradan sonrası pyomo fonksiyonundakiyle aynı mantık – sadece .X kullan)
    # overlap kontrolleri için de:
    # ops_m = [(S[i].X, C[i].X, i) ...] vb.

    print("================= END SOLUTION CHECK (GUROBI) =================\n")
