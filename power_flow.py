import numpy as np
import time
from scipy.optimize import root

# --- 1. Definição dos Dados do Sistema ---
# Dados baseados nas Tabelas 1 e 2 do documento do Trabalho

BASE_MVA = 100.0  # Base de potência
NUM_BUSES = 8  # Sistema de 8 barras

# Tolerância de convergência e número máximo de iterações
TOLERANCE = 1e-5
MAX_ITER = 200


def get_line_data():
    """
    Retorna os dados das linhas (Tabela 1).
    As barras 'de' e 'para' são mantidas em 1-based para clareza,
    mas serão convertidas para 0-based na construção da Ybus.
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
    'V' é a magnitude da tensão especificada para barras Slack/PV.
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
# Seguindo as Questões 1 a 7 do documento


def build_ybus(line_data, num_buses):
    """
    Constrói a matriz de admitância Ybus.
    """
    # Inicializa uma matriz Ybus 8x8 com números complexos
    ybus = np.zeros((num_buses, num_buses), dtype=complex)

    # Processa cada linha de transmissão
    for line in line_data:
        # Converte de 1-based (documento) para 0-based (Python)
        i = line["de"] - 1
        j = line["para"] - 1

        # Questão 2: Impedância de linha Z = R + jX
        z_linha = line["R"] + 1j * line["X"]

        # Questão 3: Admitância de linha y_ij = 1/Z
        y_linha = 1.0 / z_linha

        # Susceptância de carga (shunt), Bsh. Dividida por 2 (modelo PI).
        # "somar metade da susceptância de carga... conectada à barra"
        y_sh_half = 1j * line["Bsh"] / 2.0

        # Questão 4: Preenche elementos fora da diagonal Y_ij = -y_ij
        ybus[i, j] -= y_linha
        ybus[j, i] -= y_linha  # Matriz simétrica

        # Questão 5 e 7: Preenche elementos da diagonal
        # Y_ii = soma(y_ij) + soma(y_sh_i)
        ybus[i, i] += y_linha + y_sh_half
        ybus[j, j] += y_linha + y_sh_half

    print("--- Matriz Ybus (p.u.) ---")
    # Imprime a matriz formatada para melhor visualização
    for row in ybus:
        print(" ".join(f"{val.real:+.4f}{val.imag:+.4f}j" for val in row))
    print("-" * 28 + "\n")
    return ybus


# --- 3. Preparação dos Vetores para Cálculo ---


def get_initial_state(bus_data):
    """
    Prepara os vetores de estado inicial (flat start) e valores especificados.
    """
    num_buses = len(bus_data)

    # V_mag_spec: Vetor de magnitudes de tensão especificadas (para Slack/PV)
    # V_ang_spec: Vetor de ângulos de tensão especificados (para Slack)
    # P_spec: Potência ativa líquida injetada (Pg - Pd)
    # Q_spec: Potência reativa líquida injetada (Qg - Qd)
    # bus_types: Mapeamento do tipo de cada barra

    V_mag_spec = np.zeros(num_buses)
    V_ang_spec = np.zeros(num_buses)
    P_spec = np.zeros(num_buses)
    Q_spec = np.zeros(num_buses)
    bus_types = []

    # Vetor de tensão inicial (V_flat) para o 'flat start'
    # Barras PQ e PV começam com 1.0 p.u. e ângulo 0
    V_flat = np.ones(num_buses, dtype=complex)

    for i, bus in enumerate(bus_data):
        bus_types.append(bus["type"])

        # Potências especificadas (Pg - Pd) e (Qg - Qd)
        P_spec[i] = bus["Pg"] - bus["Pd"]
        Q_spec[i] = bus["Qg"] - bus["Qd"]

        if bus["type"] == "Slack":
            V_mag_spec[i] = bus["V"]
            V_ang_spec[i] = np.radians(bus["delta"])
            V_flat[i] = V_mag_spec[i] * np.exp(1j * V_ang_spec[i])

        elif bus["type"] == "PV":
            V_mag_spec[i] = bus["V"]
            # Mantém V_flat[i] em 1.0 p.u. (ou V_spec[i]) com ângulo 0
            V_flat[i] = V_mag_spec[i] * np.exp(1j * 0.0)

        elif bus["type"] == "PQ":
            # Mantém V_flat[i] em 1.0 p.u. com ângulo 0
            pass  # V_flat[i] já é 1.0 + 0j

    return V_flat, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types


# --- 4. Solucionador: Gauss-Jacobi ---


def solve_gauss_jacobi(ybus, V_initial, P_spec, Q_spec, V_mag_spec, bus_types):
    """
    Calcula o fluxo de potência usando o método de Gauss-Jacobi.
    """
    V_k = V_initial.copy()  # V na iteração k
    V_k1 = V_k.copy()  # V na iteração k+1
    num_buses = len(V_k)

    start_time = time.time()

    for iter in range(MAX_ITER):
        # O método de Jacobi calcula todos os novos valores de V_k1
        # usando APENAS os valores de V_k.

        for i in range(num_buses):
            if bus_types[i] == "Slack":
                continue  # Tensão da barra Slack é fixa

            # Cálculo do somatório Y_ij * V_j (para j != i)
            # Usando os valores da iteração anterior (V_k)
            YV_sum = np.dot(ybus[i, :], V_k) - ybus[i, i] * V_k[i]

            if bus_types[i] == "PQ":
                # Barra PQ: V e delta são desconhecidos
                # V_i(k+1) = (1/Y_ii) * [ (P_i_spec - j*Q_i_spec) / V_i(k)* - soma(Y_ij * V_j(k)) ]
                S_spec = P_spec[i] - 1j * Q_spec[i]
                V_k1[i] = (1 / ybus[i, i]) * (S_spec / np.conj(V_k[i]) - YV_sum)

            elif bus_types[i] == "PV":
                # Barra PV: delta e Q são desconhecidos. V é fixo.

                # 1. Calcular Q_i(k+1) usando V_i(k)
                # S_i = V_i * I_i* = V_i * conj(Y_ii*V_i + soma(Y_ij*V_j))
                S_calc_k = V_k[i] * np.conj(ybus[i, i] * V_k[i] + YV_sum)
                Q_calc = S_calc_k.imag
                # (Aqui poderiam ser aplicados limites de Q, mas não foram dados)

                # 2. Calcular V_i(k+1) como se fosse uma barra PQ, usando P_spec e Q_calc
                S_spec_temp = P_spec[i] - 1j * Q_calc
                V_temp = (1 / ybus[i, i]) * (S_spec_temp / np.conj(V_k[i]) - YV_sum)

                # 3. Corrigir a magnitude de V_temp para V_spec, mantendo o novo ângulo
                V_k1[i] = V_mag_spec[i] * (V_temp / np.abs(V_temp))

        # Verificação de convergência (diferença entre V_k1 e V_k)
        mismatch = np.max(np.abs(V_k1 - V_k))
        if mismatch < TOLERANCE:
            end_time = time.time()
            return V_k1, iter + 1, (end_time - start_time)

        # Atualiza V_k para a próxima iteração
        V_k = V_k1.copy()

    # Se não convergiu
    end_time = time.time()
    return V_k, MAX_ITER, (end_time - start_time)


# --- 5. Solucionador: Gauss-Seidel ---


def solve_gauss_seidel(ybus, V_initial, P_spec, Q_spec, V_mag_spec, bus_types):
    """
    Calcula o fluxo de potência usando o método de Gauss-Seidel.
    """
    V = V_initial.copy()
    num_buses = len(V)

    start_time = time.time()

    for iter in range(MAX_ITER):
        V_old_iter = V.copy()  # Salva V no início da iteração para checar convergência

        # O método de Seidel usa os valores mais recentes disponíveis.
        # Ao calcular V[i], ele usa os novos V[0...i-1] e os antigos V[i+1...N]

        for i in range(num_buses):
            if bus_types[i] == "Slack":
                continue  # Tensão da barra Slack é fixa

            # O np.dot(ybus[i, :], V) já usa os valores atualizados de V[0...i-1]
            YV_sum = np.dot(ybus[i, :], V) - ybus[i, i] * V[i]

            if bus_types[i] == "PQ":
                # Barra PQ: V e delta são desconhecidos
                # V_i(k+1) = (1/Y_ii) * [ (P_i_spec - j*Q_i_spec) / V_i(k)* - soma(...) ]
                S_spec = P_spec[i] - 1j * Q_spec[i]
                V[i] = (1 / ybus[i, i]) * (S_spec / np.conj(V[i]) - YV_sum)

            elif bus_types[i] == "PV":
                # Barra PV: delta e Q são desconhecidos. V é fixo.

                # 1. Calcular Q_i(k+1) usando V mais recente
                S_calc = V[i] * np.conj(ybus[i, i] * V[i] + YV_sum)
                Q_calc = S_calc.imag

                # 2. Calcular V_i(k+1) temporário
                S_spec_temp = P_spec[i] - 1j * Q_calc
                V_temp = (1 / ybus[i, i]) * (S_spec_temp / np.conj(V[i]) - YV_sum)

                # 3. Corrigir a magnitude de V_temp para V_spec, mantendo o novo ângulo
                V[i] = V_mag_spec[i] * (V_temp / np.abs(V_temp))

        # Verificação de convergência
        mismatch = np.max(np.abs(V - V_old_iter))
        if mismatch < TOLERANCE:
            end_time = time.time()
            return V, iter + 1, (end_time - start_time)

    # Se não convergiu
    end_time = time.time()
    return V, MAX_ITER, (end_time - start_time)


# --- 6. Solucionador: Newton-Raphson ---


def solve_newton_raphson(
    ybus, V_initial, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types
):
    """
    Calcula o fluxo de potência usando o método de Newton-Raphson.
    Para simplificar a implementação e evitar a construção manual da
    matriz Jacobiana, usamos a função 'root' do Scipy, que
    estima numericamente a Jacobiana e resolve o sistema.
    """

    # Índices das barras (0-based)
    pv_idx = [i for i, t in enumerate(bus_types) if t == "PV"]
    pq_idx = [i for i, t in enumerate(bus_types) if t == "PQ"]

    # O vetor de estado 'x' conterá as variáveis desconhecidas:
    # 7 ângulos (para barras PV e PQ, ou seja, 2 a 8)
    # 5 magnitudes V (para barras PQ, ou seja, 3, 4, 5, 6, 8)
    # Total = 12 variáveis

    # Índices das barras que NÃO são Slack (PV e PQ)
    non_slack_idx = pv_idx + pq_idx
    non_slack_idx.sort()  # Garante a ordem [1, 2, 3, 4, 5, 6, 7]

    # Criamos a função de mismatch (resíduos) que o 'scipy.root' tentará zerar
    def calculate_mismatch(x):
        # x é o vetor de estado atual:
        # x[0:7] = ângulos (delta) para barras 2-8
        # x[7:12] = magnitudes (V) para barras 3, 4, 5, 6, 8

        # 1. Reconstruir o vetor de tensão complexo 'V' a partir do estado 'x'
        V_mag = V_mag_spec.copy()  # Começa com valores de V_spec (PV/Slack)
        V_ang = V_ang_spec.copy()  # Começa com ângulos de Slack

        # Atualiza ângulos das barras PV e PQ (barras 2-8, índices 1-7)
        for i, bus_idx in enumerate(non_slack_idx):
            V_ang[bus_idx] = x[i]

        # Atualiza magnitudes das barras PQ (barras 3,4,5,6,8, índices 2,3,4,5,7)
        for i, bus_idx in enumerate(pq_idx):
            V_mag[bus_idx] = x[i + len(non_slack_idx)]

        # Cria V complexo
        V = V_mag * np.exp(1j * V_ang)

        # 2. Calcular S_calc = P_calc + j*Q_calc para todas as barras
        # S_i = V_i * I_i* = V_i * conj(Ybus * V)_i
        S_calc = V * np.conj(ybus.dot(V))

        # 3. Calcular Mismatches (Delta_P e Delta_Q)
        # Mismatch = Spec - Calc
        mismatch_P = P_spec - S_calc.real
        mismatch_Q = Q_spec - S_calc.imag

        # 4. Construir o vetor de mismatch 'F' que 'root' deve zerar
        # F consiste em:
        # - Mismatches de P para todas as barras NÃO-Slack (PV e PQ)
        # - Mismatches de Q para todas as barras PQ
        F_P = mismatch_P[non_slack_idx]
        F_Q = mismatch_Q[pq_idx]

        return np.concatenate((F_P, F_Q))

    # --- Execução do Newton-Raphson ---
    start_time = time.time()

    # Chute inicial (Flat Start) para as variáveis
    # Ângulos = 0.0 para barras 2-8
    x0_ang = np.zeros(len(non_slack_idx))
    # Magnitudes = 1.0 para barras PQ
    x0_V = np.ones(len(pq_idx))
    x0 = np.concatenate((x0_ang, x0_V))

    # Chama o solucionador 'root'. 'lm' (Levenberg-Marquardt) é robusto.
    sol = root(calculate_mismatch, x0, method="lm", tol=TOLERANCE)

    end_time = time.time()

    if not sol.success:
        print("AVISO: Newton-Raphson não convergiu.")
        return V_initial, MAX_ITER, (end_time - start_time)

    # Reconstruir o vetor final de tensão 'V' a partir da solução 'sol.x'
    x_final = sol.x
    V_mag_final = V_mag_spec.copy()
    V_ang_final = V_ang_spec.copy()

    for i, bus_idx in enumerate(non_slack_idx):
        V_ang_final[bus_idx] = x_final[i]
    for i, bus_idx in enumerate(pq_idx):
        V_mag_final[bus_idx] = x_final[i + len(non_slack_idx)]

    V_final = V_mag_final * np.exp(1j * V_ang_final)

    return (
        V_final,
        sol.nfev,
        (end_time - start_time),
    )  # nfev = número de avaliações da função


# --- 7. Cálculo Pós-Convergência (Potências e Fluxos) ---


def calculate_powers_and_flows(V, ybus, line_data, bus_data, base_mva):
    """
    Calcula a geração na barra Slack, a geração reativa nas barras PV,
    os fluxos nas linhas e as perdas.
    """

    # 1. Calcular potências injetadas em TODAS as barras
    S_injetada = V * np.conj(ybus.dot(V))

    print("\n--- Resultados Finais (Tensões e Potências) ---")
    print("Barra | Tensão (p.u.) | Ângulo (graus) | P Gerado (MW) | Q Gerado (MVAr)")
    print("-" * 70)

    for i in range(len(V)):
        V_mag = np.abs(V[i])
        V_ang_deg = np.angle(V[i], deg=True)
        bus = bus_data[i]

        # Potência gerada = Potência Injetada + Potência de Carga
        Pg = (S_injetada[i].real + bus["Pd"]) * base_mva
        Qg = (S_injetada[i].imag + bus["Qd"]) * base_mva

        print(
            f" {i+1:^4} |    {V_mag:^10.4f} |   {V_ang_deg:^12.4f} |   {Pg:^13.3f} |   {Qg:^14.3f}"
        )

    # 2. Calcular Fluxos nas Linhas e Perdas
    print("\n--- Fluxos e Perdas nas Linhas de Transmissão ---")
    print("  Linha  | Fluxo 'de->para' (MVA) | Fluxo 'para->de' (MVA) | Perdas (MVA)")
    print(
        " (de-para) |   P (MW)    Q (MVAr) |   P (MW)    Q (MVAr) |  P (MW)   Q (MVAr)"
    )
    print("-" * 80)

    total_perdas_P = 0.0
    total_perdas_Q = 0.0

    for line in line_data:
        # Índices 0-based
        i = line["de"] - 1
        j = line["para"] - 1

        # Admitância e Shunt da linha (mesmo cálculo da Ybus)
        z_linha = line["R"] + 1j * line["X"]
        y_linha = 1.0 / z_linha
        y_sh_half = 1j * line["Bsh"] / 2.0

        # Corrente de i para j (I_ij) e de j para i (I_ji)
        # I_ij = (Vi - Vj) * y_linha + Vi * y_sh_half
        I_ij = (V[i] - V[j]) * y_linha + V[i] * y_sh_half
        I_ji = (V[j] - V[i]) * y_linha + V[j] * y_sh_half

        # Potência S_ij = Vi * I_ij*
        S_ij = V[i] * np.conj(I_ij) * base_mva
        S_ji = V[j] * np.conj(I_ji) * base_mva

        # Perdas na linha = S_ij + S_ji
        S_perdas = S_ij + S_ji
        total_perdas_P += S_perdas.real
        total_perdas_Q += S_perdas.imag

        print(
            f"  ({i+1:^2}-{j+1:^2}) | {S_ij.real:^9.2f} {S_ij.imag:^9.2f} | {S_ji.real:^9.2f} {S_ji.imag:^9.2f} | {S_perdas.real:^7.2f} {S_perdas.imag:^9.2f}"
        )

    print("-" * 80)
    print(f"Perdas Totais: {total_perdas_P:^39.2f} MW {total_perdas_Q:^9.2f} MVAr")


# --- 8. Função Principal de Execução ---


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

    print("Iniciando Solução por Newton-Raphson...")
    V_nr, iter_nr, time_nr = solve_newton_raphson(
        ybus, V_flat, P_spec, Q_spec, V_mag_spec, V_ang_spec, bus_types
    )
    print(
        f"Newton-Raphson convergiu em {iter_nr} iterações (avaliações) e {time_nr:.6f} segundos."
    )
    calculate_powers_and_flows(V_nr, ybus, line_data, bus_data, BASE_MVA)

    print("\n" + "=" * 80 + "\n")
    print("--- Análise de Convergência (Comparação) ---")
    print(f"Método            | Iterações | Tempo (s)")
    print("-" * 42)
    print(f"Gauss-Jacobi      | {iter_j:^9} | {time_j:^10.6f}")
    print(f"Gauss-Seidel      | {iter_s:^9} | {time_s:^10.6f}")
    print(f"Newton-Raphson    | {iter_nr:^9} | {time_nr:^10.6f}")


if __name__ == "__main__":

    main()
