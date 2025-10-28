import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Dados das linhas (R, X, Bsh) e montagem da Ybus
# ============================================================
dados_linhas = [
    (1,2,0.0,0.1730,0.00865),
    (1,8,0.0,0.1584,0.00792),
    (2,3,0.0,0.1248,0.00624),
    (2,4,0.0,0.0720,0.00360),
    (3,4,0.0,0.1200,0.00600),
    (3,5,0.0,0.0816,0.00408),
    (4,5,0.0,0.1152,0.00576),
    (5,6,0.0,0.1200,0.00600),
    (6,7,0.0,0.1056,0.00528),
    (7,8,0.0,0.0864,0.00432)
]

# Função para montar matriz admitância Ybus
def montar_Ybus(dados_linhas, n_barras):
    Ybus = np.zeros((n_barras, n_barras), dtype=complex)
    for de, para, r, x, bsh in dados_linhas:
        z = complex(r, x)
        y = 1/z
        b = complex(0, bsh/2)
        # Elementos fora da diagonal
        Ybus[de-1, para-1] -= y
        Ybus[para-1, de-1] -= y
        # Elementos da diagonal
        Ybus[de-1, de-1] += y + b
        Ybus[para-1, para-1] += y + b
    return Ybus

Ybus = montar_Ybus(dados_linhas, 8)

# ============================================================
# 2. Dados das barras
# ============================================================
barras = [
    {"tipo":"Slack","Pg":0,"Qg":0,"Pd":0,"Qd":0,"V":1.05},
    {"tipo":"PV","Pg":0.20,"Qg":0,"Pd":0.10,"Qd":0.05,"V":1.05},
    {"tipo":"PQ","Pg":0,"Qg":0,"Pd":0.50,"Qd":0.20,"V":1.0},
    {"tipo":"PQ","Pg":0,"Qg":0,"Pd":0.60,"Qd":0.25,"V":1.0},
    {"tipo":"PQ","Pg":0,"Qg":0,"Pd":0.40,"Qd":0.15,"V":1.0},
    {"tipo":"PQ","Pg":0,"Qg":0,"Pd":0.30,"Qd":0.10,"V":1.0},
    {"tipo":"PV","Pg":0.30,"Qg":0,"Pd":0,"Qd":0,"V":1.04},
    {"tipo":"PQ","Pg":0,"Qg":0,"Pd":0.60,"Qd":0.20,"V":1.0}
]

# ============================================================
# 3. Método Gauss–Jacobi
# ============================================================
def gauss_jacobi(Ybus, barras, tol=1e-6, max_iter=100):
    n = len(barras)
    V = np.array([b["V"] for b in barras], dtype=complex)
    historico = []
    mismatch_hist = []

    for it in range(max_iter):
        V_prev = V.copy()
        V_new = V.copy()   # nova iteração calculada inteiramente a partir de V_prev

        for i, barra in enumerate(barras):
            if barra["tipo"] == "Slack":
                continue

            P = barra["Pg"] - barra["Pd"]
            Q = barra["Qg"] - barra["Qd"] if barra["tipo"] == "PQ" else 0
            S = complex(P, Q)

            # Agora usa apenas V_prev
            soma = sum(Ybus[i, j] * V_prev[j] for j in range(n) if j != i)

            V_new[i] = (1 / Ybus[i, i]) * ((np.conj(S) / np.conj(V_prev[i])) - soma)

            if barra["tipo"] == "PV":
                V_new[i] = barra["V"] * np.exp(1j * np.angle(V_new[i]))

        V = V_new.copy()
        historico.append((it+1, abs(V), np.degrees(np.angle(V))))
        mismatch_hist.append(np.linalg.norm(V - V_prev, np.inf))

        if mismatch_hist[-1] < tol:
            return V, it+1, historico, mismatch_hist

    return V, max_iter, historico, mismatch_hist


# ============================================================
# 4. Método Gauss–Seidel
# ============================================================
def gauss_seidel(Ybus, barras, tol=1e-6, max_iter=100):
    n = len(barras)
    V = np.array([b["V"] for b in barras], dtype=complex)
    historico = []
    mismatch_hist = []

    for it in range(max_iter):
        V_prev = V.copy()
        for i, barra in enumerate(barras):
            if barra["tipo"] == "Slack":
                continue

            P = barra["Pg"] - barra["Pd"]
            Q = barra["Qg"] - barra["Qd"] if barra["tipo"] == "PQ" else 0
            S = complex(P, Q)

            # Separa termos com V[j] já atualizado (j<i) e ainda não atualizado (j>i)
            soma1 = sum(Ybus[i, j] * V[j] for j in range(i))        # já atualizado
            soma2 = sum(Ybus[i, j] * V_prev[j] for j in range(i+1, n))  # ainda da iteração anterior

            V[i] = (1 / Ybus[i, i]) * ((np.conj(S) / np.conj(V[i])) - soma1 - soma2)

            if barra["tipo"] == "PV":
                V[i] = barra["V"] * np.exp(1j * np.angle(V[i]))

        historico.append((it+1, abs(V), np.degrees(np.angle(V))))
        mismatch_hist.append(np.linalg.norm(V - V_prev, np.inf))

        if mismatch_hist[-1] < tol:
            return V, it+1, historico, mismatch_hist

    return V, max_iter, historico, mismatch_hist

# ============================================================
# 5. Método Newton–Raphson (completo)
# ============================================================
def newton_raphson(Ybus, barras, tol=1e-6, max_iter=20):
    n = len(barras)
    V = np.array([b["V"] for b in barras], dtype=complex)

    # Índices das barras
    pq = [i for i,b in enumerate(barras) if b["tipo"]=="PQ"]
    pv = [i for i,b in enumerate(barras) if b["tipo"]=="PV"]
    mismatch_hist = []

    for it in range(max_iter):
        # Cálculo das potências injetadas
        Pcalc = np.zeros(n)
        Qcalc = np.zeros(n)
        for i in range(n):
            soma = 0
            for j in range(n):
                soma += V[i] * np.conj(Ybus[i,j] * V[j])
            Pcalc[i] = soma.real
            Qcalc[i] = soma.imag

        # Vetor de mismatch (ΔP, ΔQ)
        dP, dQ = [], []
        for i in range(n):
            barra = barras[i]
            if barra["tipo"] == "Slack":
                continue
            Pesp = barra["Pg"] - barra["Pd"]
            Qesp = barra["Qg"] - barra["Qd"]
            dP.append(Pesp - Pcalc[i])
            if barra["tipo"] == "PQ":
                dQ.append(Qesp - Qcalc[i])
        mismatch = np.array(dP + dQ)
        mismatch_hist.append(np.max(abs(mismatch)))

        # Critério de convergência
        if np.max(abs(mismatch)) < tol:
            return V, it+1, mismatch_hist

        # Montagem do Jacobiano completo
        ang = np.angle(V)
        Vm = abs(V)
        npq = len(pq)
        nvar = (n-1) + npq
        J = np.zeros((nvar,nvar))

        # Matrizes H (dP/dθ)
        for i in range(1,n):
            mi = i
            for k in range(1,n):
                mk = k
                if mi==mk:
                    J[i-1,k-1] = -Qcalc[mi] - (Vm[mi]**2)*Ybus[mi,mi].imag
                else:
                    J[i-1,k-1] = Vm[mi]*Vm[mk]*(Ybus[mi,mk].real*np.sin(ang[mi]-ang[mk]) -
                                               Ybus[mi,mk].imag*np.cos(ang[mi]-ang[mk]))
        # N
        colN = n-1
        for i_idx,i in enumerate(range(1,n)):
            mi = i
            for k_idx,k in enumerate(pq):
                if mi==k:
                    J[i_idx,colN+k_idx] = Pcalc[mi]/Vm[mi] + Ybus[mi,mi].real*Vm[mi]
                else:
                    J[i_idx,colN+k_idx] = Vm[mi]*(Ybus[mi,k].real*np.cos(ang[mi]-ang[k]) +
                                                  Ybus[mi,k].imag*np.sin(ang[mi]-ang[k]))
        # M
        rowM = n-1
        for i_idx,i in enumerate(pq):
            mi = i
            for k in range(1,n):
                mk = k
                if mi==mk:
                    J[rowM+i_idx,k-1] = Pcalc[mi] - (Vm[mi]**2)*Ybus[mi,mi].real
                else:
                    J[rowM+i_idx,k-1] = -Vm[mi]*Vm[mk]*(Ybus[mi,mk].real*np.cos(ang[mi]-ang[mk]) +
                                                        Ybus[mi,mk].imag*np.sin(ang[mi]-ang[mk]))
        # L
        for i_idx,i in enumerate(pq):
            mi = i
            for k_idx,k in enumerate(pq):
                if mi==k:
                    J[rowM+i_idx,colN+k_idx] = Qcalc[mi]/Vm[mi] - Ybus[mi,mi].imag*Vm[mi]
                else:
                    J[rowM+i_idx,colN+k_idx] = Vm[mi]*(Ybus[mi,k].real*np.sin(ang[mi]-ang[k]) -
                                                       Ybus[mi,k].imag*np.cos(ang[mi]-ang[k]))

        # Resolve sistema linear
        dV = np.linalg.solve(J,mismatch[:nvar])
        dth = dV[:n-1]
        dVm = dV[n-1:]

        # Atualiza variáveis de estado
        ang[1:] += dth
        for i,k in enumerate(pq):
            Vm[k] += dVm[i]
        V = Vm*np.exp(1j*ang)

    return V, max_iter, mismatch_hist

# ============================================================
# 6. Comparação dos métodos
# ============================================================
Vj, itj, _, hist_j = gauss_jacobi(Ybus, barras)
Vs, its, _, hist_s = gauss_seidel(Ybus, barras)
Vn, itn, hist_n = newton_raphson(Ybus, barras)

print("Gauss-Jacobi convergiu em", itj, "iterações")
for i,v in enumerate(Vj,1):
    print(f"Barra {i}: |V|={abs(v):.4f}, ∠={np.degrees(np.angle(v)):.2f}°")
    
print("Gauss-Seidel convergiu em", its, "iterações")
for i,v in enumerate(Vs,1):
    print(f"Barra {i}: |V|={abs(v):.4f}, ∠={np.degrees(np.angle(v)):.2f}°")

print("Newton-Raphson convergiu em", itn, "iterações")
for i,v in enumerate(Vn,1):
    print(f"Barra {i}: |V|={abs(v):.4f}, ∠={np.degrees(np.angle(v)):.2f}°")

# ============================================================
# 7. Gráfico de convergência
# ============================================================
plt.figure(figsize=(8,6))
plt.semilogy(range(1,len(hist_j)+1), hist_j, label="Gauss-Jacobi")
plt.semilogy(range(1,len(hist_s)+1), hist_s, label="Gauss-Seidel")
plt.semilogy(range(1,len(hist_n)+1), hist_n, label="Newton-Raphson")
plt.xlabel("Iterações")
plt.ylabel("Mismatch máximo")
plt.title("Convergência dos Métodos de Fluxo de Potência")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()