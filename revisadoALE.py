import numpy as np
import time
from scipy.optimize import root

# --- 1. Definição dos Dados do Sistema ---
BASE_MVA = 100.0
NUM_BUSES = 8
TOLERANCE = 1e-5
MAX_ITER = 200


def get_line_data():
    """
    Retorna os dados das linhas (Tabela 1).
    """
    line_data = [
        # De, Para, R (p.u.), X (p.u.), Bsh (p.u.)
        {"de": 1, "para": 2, "R": 0.0, "X": 0.1730, "Bsh": 0.00865},
        {"de": 1, "para": 8, "R": 0.0, "X": 0.1584, "Bsh": 0.00792},
        {"de": 2, "para": 3, "R": 0.0, "X": 0.1248, "Bsh": 0.00624},
        {"de": 2, "para": 4, "R": 0.0, "X": 0.0720, "Bsh": 0.00360},
        {"de": 3, "para": 4, "R": 0.0, "X": 0.1200, "Bsh": 0.00600},
        {"de": 3, "para": 5, "R": 0.0, "X": 0.0816, "Bsh": 0.00408},
        {"de": 4, "para": 5, "R": 0.0, "X": 0.1152, "Bsh": 0.00576},
        {"de": 4, "para": 6, "R": 0.0, "X": 0.1200, "Bsh": 0.00600},
        {"de": 5, "para": 7, "R": 0.0, "X": 0.1056, "Bsh": 0.00528},
        {"de": 7, "para": 8, "R": 0.0, "X": 0.0864, "Bsh": 0.00432},
    ]
    return line_data


def get_bus_data():
    """
    Retorna os dados das barras (Tabela 2).
    """
    bus_data = [
        # Barra, Tipo, Pg, Qg, Pd, Qd, V, Angulo (graus)
        {
            "bus": 1,
            "type": "Slack",
            "Pg": 0.0,
            "Qg": 0.0,
            "Pd": 0.0,
            "Qd": 0.0,
            "V": 1.05,
            "delta": 0.0,
        },
        {
            "bus": 2,
            "type": "PV",
            "Pg": 0.2,
            "Qg": 0.0,
            "Pd": 0.0,
            "Qd": 0.0,
            "V": 1.05,
            "delta": None,
        },
        {
            "bus": 3,
            "type": "PQ",
            "Pg": 0.0,
            "Qg": 0.0,
            "Pd": 0.5,
            "Qd": 0.2,
            "V": None,
            "delta": None,
        },
        {
            "bus": 4,
            "type": "PQ",
            "Pg": 0.0,
            "Qg": 0.0,
            "Pd": 0.6,
            "Qd": 0.3,
            "V": None,
            "delta": None,
        },
        {
            "bus": 5,
            "type": "PQ",
            "Pg": 0.0,
            "Qg": 0.0,
            "Pd": 0.7,
            "Qd": 0.4,
            "V": None,
            "delta": None,
        },
        {
            "bus": 6,
            "type": "PQ",
            "Pg": 0.0,
            "Qg": 0.0,
            "Pd": 0.8,
            "Qd": 0.5,
            "V": None,
            "delta": None,
        },
        {
            "bus": 7,
            "type": "PV",
            "Pg": 0.3,
            "Qg": 0.0,
            "Pd": 0.0,
            "Qd": 0.0,
            "V": 1.04,
            "delta": None,
        },
        {
            "bus": 8,
            "type": "PQ",
            "Pg": 0.0,
            "Qg": 0.0,
            "Pd": 0.4,
            "Qd": 0.2,
            "V": None,
            "delta": None,
        },
    ]
    return bus_data


# --- 2. Construção da Matriz de Admitância (Ybus) ---
# (O mesmo de antes)
def build_ybus(line_data, num_buses):
    ybus = np.zeros((num_buses, num_buses), dtype=complex)
    for line in line_data:
        i = line["de"] - 1
        j = line["para"] - 1
        z_linha = line["R"] + 1j * line["X"]
        y_linha = 1.0 / z_linha
        y_sh_half = 1j * line["Bsh"] / 2.0
        ybus[i, j] -= y_linha
        ybus[j, i] -= y_linha
        ybus[i, i] += y_linha + y_sh_half
        ybus[j, j] += y_linha + y_sh_half

    print("--- Matriz Ybus (p.u.) ---")
    for row in ybus:
        print(" ".join(f"{val.real:+.4f}{val.imag:+.4f}j" for val in row))
    print("-" * 28 + "\n")
    return ybus


# --- 3. Preparação dos Vetores para Cálculo ---
# (O mesmo de antes)
def get_initial_state(bus_data):
    num_buses = len(bus_data)
    V_mag_spec = np.zeros(num_buses)
    V_ang_spec = np.zeros(num_buses)
    P_spec = np.zeros(num_buses)
    Q_spec = np.zeros(num_buses)
    bus_types = []
    V_flat = np.ones(num_buses, dtype=complex)

    for i, bus in enumerate(bus_data):
        bus_types.append(bus["type"])
        P_spec[i] = bus["Pg"] - bus["Pd"]
        Q_spec[i] = bus["Qg"] - bus["Qd"]

        if bus["type"] == "Slack":
            V_mag_spec[i] = bus["V"]
            V_ang_spec[i] = np.radians(bus["delta"])
            V_flat[i] = V_mag_spec[i] * np.exp(1j * V_ang_spec[i])
        elif bus["type"] == "PV":
            V_mag_spec[i] = bus["V"]
            V_flat[i] = V_mag_spec[i] * np.exp(1j * 0.0)
        elif bus["type"] == "PQ":
            pass

    return V_flat, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types


# --- 4. Solucionador: Gauss-Jacobi ---
# (O mesmo de antes)
def solve_gauss_jacobi(ybus, V_initial, P_spec, Q_spec, V_mag_spec, bus_types):
    V_k = V_initial.copy()
    V_k1 = V_k.copy()
    num_buses = len(V_k)
    start_time = time.time()

    for iter in range(MAX_ITER):
        for i in range(num_buses):
            if bus_types[i] == "Slack":
                continue
            YV_sum = np.dot(ybus[i, :], V_k) - ybus[i, i] * V_k[i]
            if bus_types[i] == "PQ":
                S_spec = P_spec[i] - 1j * Q_spec[i]
                V_k1[i] = (1 / ybus[i, i]) * (S_spec / np.conj(V_k[i]) - YV_sum)
            elif bus_types[i] == "PV":
                S_calc_k = V_k[i] * np.conj(ybus[i, i] * V_k[i] + YV_sum)
                Q_calc = S_calc_k.imag
                S_spec_temp = P_spec[i] - 1j * Q_calc
                V_temp = (1 / ybus[i, i]) * (S_spec_temp / np.conj(V_k[i]) - YV_sum)
                V_k1[i] = V_mag_spec[i] * (V_temp / np.abs(V_temp))

        mismatch = np.max(np.abs(V_k1 - V_k))
        if mismatch < TOLERANCE:
            end_time = time.time()
            return V_k1, iter + 1, (end_time - start_time)
        V_k = V_k1.copy()

    end_time = time.time()
    return V_k, MAX_ITER, (end_time - start_time)


# --- 5. Solucionador: Gauss-Seidel ---
# (O mesmo de antes)
def solve_gauss_seidel(ybus, V_initial, P_spec, Q_spec, V_mag_spec, bus_types):
    V = V_initial.copy()
    num_buses = len(V)
    start_time = time.time()

    for iter in range(MAX_ITER):
        V_old_iter = V.copy()
        for i in range(num_buses):
            if bus_types[i] == "Slack":
                continue
            YV_sum = np.dot(ybus[i, :], V) - ybus[i, i] * V[i]
            if bus_types[i] == "PQ":
                S_spec = P_spec[i] - 1j * Q_spec[i]
                V[i] = (1 / ybus[i, i]) * (S_spec / np.conj(V[i]) - YV_sum)
            elif bus_types[i] == "PV":
                S_calc = V[i] * np.conj(ybus[i, i] * V[i] + YV_sum)
                Q_calc = S_calc.imag
                S_spec_temp = P_spec[i] - 1j * Q_calc
                V_temp = (1 / ybus[i, i]) * (S_spec_temp / np.conj(V[i]) - YV_sum)
                V[i] = V_mag_spec[i] * (V_temp / np.abs(V_temp))

        mismatch = np.max(np.abs(V - V_old_iter))
        if mismatch < TOLERANCE:
            end_time = time.time()
            return V, iter + 1, (end_time - start_time)

    end_time = time.time()
    return V, MAX_ITER, (end_time - start_time)


# --- 6. Solucionador: Newton-Raphson ---
# *** FUNÇÃO MODIFICADA ***
def solve_newton_raphson(
    ybus, V_initial, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types
):
    """
    Calcula o fluxo de potência usando o método de Newton-Raphson CLÁSSICO.
    Constrói a Jacobiana numericamente (por diferenças finitas)
    e resolve o sistema linear a cada iteração.
    """
    start_time = time.time()

    # --- 1. Setup (idem ao anterior) ---
    pv_idx = [i for i, t in enumerate(bus_types) if t == "PV"]
    pq_idx = [i for i, t in enumerate(bus_types) if t == "PQ"]
    non_slack_idx = pv_idx + pq_idx
    non_slack_idx.sort()  # Garante [1, 2, 3, 4, 5, 6, 7]

    # Número de variáveis = 7 ângulos (PV+PQ) + 5 magnitudes (PQ) = 12
    num_vars = len(non_slack_idx) + len(pq_idx)

    # --- 2. Função de Mismatch (idem ao anterior) ---
    # Esta função interna será usada repetidamente
    def calculate_mismatch(x):
        # x é o vetor de estado atual:
        # x[0:7] = ângulos (delta) para barras 2-8
        # x[7:12] = magnitudes (V) para barras 3, 4, 5, 6, 8

        # 2.1. Reconstruir o vetor de tensão 'V' a partir do estado 'x'
        V_mag = V_mag_spec.copy()  # Começa com valores de V_spec (PV/Slack)
        V_ang = V_ang_spec.copy()  # Começa com ângulos de Slack

        for i, bus_idx in enumerate(non_slack_idx):
            V_ang[bus_idx] = x[i]
        for i, bus_idx in enumerate(pq_idx):
            V_mag[bus_idx] = x[i + len(non_slack_idx)]

        V = V_mag * np.exp(1j * V_ang)

        # 2.2. Calcular potências S_calc
        S_calc = V * np.conj(ybus.dot(V))

        # 2.3. Calcular Mismatches
        mismatch_P = P_spec - S_calc.real
        mismatch_Q = Q_spec - S_calc.imag

        # 2.4. Construir o vetor de mismatch 'F'
        F_P = mismatch_P[non_slack_idx]
        F_Q = mismatch_Q[pq_idx]

        return np.concatenate((F_P, F_Q))

    # --- 3. Chute Inicial (idem ao anterior) ---
    x0_ang = np.zeros(len(non_slack_idx))  # Ângulos = 0.0
    x0_V = np.ones(len(pq_idx))  # Magnitudes = 1.0
    x = np.concatenate((x0_ang, x0_V))  # Vetor de estado inicial 'x0'

    # --- 4. Loop de Iteração N-R Clássico ---
    h = 1e-6  # Perturbação pequena para Jacobiana numérica
    iter_count = 0  # Contador de iterações

    for iter_count in range(MAX_ITER):

        # 4.1. Calcular Mismatch F(x_k)
        F = calculate_mismatch(x)

        # 4.2. Checar Convergência
        mismatch = np.max(np.abs(F))
        if mismatch < TOLERANCE:
            break  # Sucesso!

        # 4.3. Construir Jacobiana J(x_k) numericamente
        J = np.zeros((num_vars, num_vars))
        for j in range(num_vars):
            x_pert = x.copy()
            x_pert[j] += h  # Perturba uma variável de estado
            F_pert = calculate_mismatch(x_pert)  # Calcula novo mismatch
            J[:, j] = (F_pert - F) / h  # Coluna 'j' da Jacobiana

        # 4.4. Resolver sistema linear J * delta_x = -F
        try:
            delta_x = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            print("AVISO: Jacobiana singular. N-R falhou.")
            end_time = time.time()
            return V_initial, MAX_ITER, (end_time - start_time)

        # 4.5. Atualizar estado x_k+1 = x_k + delta_x
        x += delta_x

    # --- 5. Fim do Loop ---
    end_time = time.time()

    if iter_count == MAX_ITER - 1 and mismatch >= TOLERANCE:
        print(f"AVISO: Newton-Raphson não convergiu após {MAX_ITER} iterações.")

    # --- 6. Reconstruir V final (idem ao anterior) ---
    V_mag_final = V_mag_spec.copy()
    V_ang_final = V_ang_spec.copy()

    for i, bus_idx in enumerate(non_slack_idx):
        V_ang_final[bus_idx] = x[i]
    for i, bus_idx in enumerate(pq_idx):
        V_ang_final[bus_idx] = x[i + len(non_slack_idx)]

    V_final = V_mag_final * np.exp(1j * V_ang_final)

    # Retorna o número de iterações (passos de atualização)
    # Se quebrou em iter_count=3, foram 3 iterações (0, 1, 2)
    return V_final, iter_count, (end_time - start_time)


# --- 7. Cálculo Pós-Convergência (Potências e Fluxos) ---
# (O mesmo de antes)
def calculate_powers_and_flows(V, ybus, line_data, bus_data, base_mva):
    S_injetada = V * np.conj(ybus.dot(V))

    print("\n--- Resultados Finais (Tensões e Potências) ---")
    print("Barra | Tensão (p.u.) | Ângulo (graus) | P Gerado (MW) | Q Gerado (MVAr)")
    print("-" * 70)

    for i in range(len(V)):
        V_mag = np.abs(V[i])
        V_ang_deg = np.angle(V[i], deg=True)
        bus = bus_data[i]
        Pg = (S_injetada[i].real + bus["Pd"]) * base_mva
        Qg = (S_injetada[i].imag + bus["Qd"]) * base_mva
        print(
            f" {i+1:^4} |    {V_mag:^10.4f} |   {V_ang_deg:^12.4f} |   {Pg:^13.3f} |   {Qg:^14.3f}"
        )

    print("\n--- Fluxos e Perdas nas Linhas de Transmissão ---")
    print("  Linha  | Fluxo 'de->para' (MVA) | Fluxo 'para->de' (MVA) | Perdas (MVA)")
    print(
        " (de-para) |   P (MW)    Q (MVAr) |   P (MW)    Q (MVAr) |  P (MW)   Q (MVAr)"
    )
    print("-" * 80)

    total_perdas_P = 0.0
    total_perdas_Q = 0.0

    for line in line_data:
        i = line["de"] - 1
        j = line["para"] - 1
        z_linha = line["R"] + 1j * line["X"]
        y_linha = 1.0 / z_linha
        y_sh_half = 1j * line["Bsh"] / 2.0
        I_ij = (V[i] - V[j]) * y_linha + V[i] * y_sh_half
        I_ji = (V[j] - V[i]) * y_linha + V[j] * y_sh_half
        S_ij = V[i] * np.conj(I_ij) * base_mva
        S_ji = V[j] * np.conj(I_ji) * base_mva
        S_perdas = S_ij + S_ji
        total_perdas_P += S_perdas.real
        total_perdas_Q += S_perdas.imag
        print(
            f"  ({i+1:^2}-{j+1:^2}) | {S_ij.real:^9.2f} {S_ij.imag:^9.2f} | {S_ji.real:^9.2f} {S_ji.imag:^9.2f} | {S_perdas.real:^7.2f} {S_perdas.imag:^9.2f}"
        )

    print("-" * 80)
    print(f"Perdas Totais: {total_perdas_P:^39.2f} MW {total_perdas_Q:^9.2f} MVAr")


# --- 8. Função Principal de Execução ---
# *** FUNÇÃO MODIFICADA ***
def main():
    # 1. Obter dados e construir Ybus
    line_data = get_line_data()
    bus_data = get_bus_data()
    ybus = build_ybus(line_data, NUM_BUSES)

    # 2. Obter vetores de estado inicial
    V_flat, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types = get_initial_state(
        bus_data
    )

    # --- 3. Resolver e Comparar os Métodos ---

    print("Iniciando Solução por Gauss-Jacobi...")
    V_jacobi, iter_j, time_j = solve_gauss_jacobi(
        ybus, V_flat, P_spec, Q_spec, V_mag_spec, bus_types
    )
    print(f"Gauss-Jacobi convergiu em {iter_j} iterações e {time_j:.6f} segundos.")
    calculate_powers_and_flows(V_jacobi, ybus, line_data, bus_data, BASE_MVA)

    print("\n" + "=" * 80 + "\n")

    print("Iniciando Solução por Gauss-Seidel...")
    V_seidel, iter_s, time_s = solve_gauss_seidel(
        ybus, V_flat, P_spec, Q_spec, V_mag_spec, bus_types
    )
    print(f"Gauss-Seidel convergiu em {iter_s} iterações e {time_s:.6f} segundos.")
    calculate_powers_and_flows(V_seidel, ybus, line_data, bus_data, BASE_MVA)

    print("\n" + "=" * 80 + "\n")

    # Chamada para a nova função N-R
    print("Iniciando Solução por Newton-Raphson...")
    V_nr, iter_nr, time_nr = solve_newton_raphson(
        ybus, V_flat, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types
    )
    # Texto do print atualizado
    print(
        f"Newton-Raphson convergiu em {iter_nr} iterações e {time_nr:.6f} segundos."
    )
    calculate_powers_and_flows(V_nr, ybus, line_data, bus_data, BASE_MVA)

    print("\n" + "=" * 80 + "\n")
    print("--- Análise de Convergência (Comparação) ---")
    print(f"Método                     | Iterações | Tempo (s)")
    print("-" * 52)
    print(f"Gauss-Jacobi               | {iter_j:^9} | {time_j:^10.6f}")
    print(f"Gauss-Seidel               | {iter_s:^9} | {time_s:^10.6f}")
    # Texto do print atualizado
    print(f"Newton-Raphson  | {iter_nr:^9} | {time_nr:^10.6f}")


if __name__ == "__main__":
    main()