# Projeto de Fluxo de Potência - SEP I

Este repositório contém a implementação em Python dos métodos de cálculo de fluxo de potência para o projeto da disciplina "SEP I - Sistemas Elétricos de Potência" do Centro Universitário SENAI CIMATEC, ministrada pelo Prof. Artur Passos Dias Lima.

## 1. Objetivo

O objetivo principal deste projeto é **calcular as tensões e os fluxos de potência (ativa e reativa) em todas as barras e linhas** de um sistema elétrico de potência.

Além disso, o projeto visa **comparar os resultados e o desempenho** de três métodos numéricos clássicos:

* Método de Gauss-Jacobi
* Método de Gauss-Seidel
* Método de Newton-Raphson

## 2. O Sistema

O sistema modelado é composto por **8 barras e 10 linhas**. Os dados de entrada (impedâncias, cargas, geração e tensões) foram fornecidos no documento do projeto e estão definidos diretamente no código.

* **Potência Base:** 100 MVA
* **Barras:** 1 Slack, 2 PV (Geração) e 5 PQ (Carga)
* **Linhas:** 10 linhas de transmissão

## 3. Requisitos e Execução

Este projeto foi desenvolvido em Python 3 e requer as seguintes bibliotecas:

* `numpy` (para cálculos matriciais e números complexos)
* `scipy` (utilizada para o solucionador do método Newton-Raphson)

**Para instalar as dependências:**

```bash
pip install numpy scipy