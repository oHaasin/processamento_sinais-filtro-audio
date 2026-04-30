"""
#! Autores: Oliver Haas, Gabriel Paiva
#* Data de última modificação: 30/04/2026
#? Objetivo: Juntar as partes do trabalho
#todo: Terminar segunda, terceira e quarta partes
"""

import carregamento_dados_1
import filtros_2
import filtragem_3
import bonus_4
import numpy as np
import matplotlib.pyplot as plt

#* Execução da primeira parte

# Abre a janela e pega o caminho do arquivo
caminho = carregamento_dados_1.carregar_wav()

# Plota o gráfico de tempo x(t) e recebe de volta a taxa e os dados do áudio
fs, dados = carregamento_dados_1.plotar_sinal_tempo(caminho)

# Usa a taxa e os dados recebidos acima para plotar X(e^jw)
carregamento_dados_1.plotar_espectro_frequencia(fs, dados)

# Abre a janela para carregar os coeficientes
numerador, denominador = carregamento_dados_1.carrega_mat()

# Passa os parâmetros e gera o gráfico de H(e^jw)
carregamento_dados_1.plotar_resposta_frequencia(numerador, denominador, fs)

# Gera a resposta ao impulso h[n] com 1000 amostras
h = carregamento_dados_1.plotar_resposta_impulso(numerador, denominador, n_amostras=1000)

#* Execução da segunda parte

# ============================================================
# 2.1 Filtragem pela equação de diferenças
# ============================================================

print("\nQuestão 2 - Filtragem por equação de diferenças...")

filtro_eqdif = filtros_2.FiltroEqDif(numerador, denominador)

y_eqdif = np.zeros_like(dados, dtype=np.float64)

for n in range(len(dados)):
    y_eqdif[n] = filtro_eqdif.filtrar_amostra(dados[n])

print("Filtragem por equação de diferenças concluída.")


# ============================================================
# 2.2 Cálculo e truncamento da resposta ao impulso
# ============================================================

N_h_total = 1000

impulso = np.zeros(N_h_total)
impulso[0] = 1.0

filtro_impulso = filtros_2.FiltroEqDif(numerador, denominador)

h = np.zeros(N_h_total)

for n in range(N_h_total):
    h[n] = filtro_impulso.filtrar_amostra(impulso[n])

h_trunc, Nh = filtros_2.truncar_resposta_impulso(h, percentual=0.01)

print("\nQuestão 2 - Resposta ao impulso:")
print("Número de amostras original =", len(h))
print("Número de amostras truncado Nh =", Nh)
print("Pico de |h[n]| =", np.max(np.abs(h)))
print("Limiar de truncagem =", 0.01 * np.max(np.abs(h)))


# Gráfico da resposta ao impulso
plt.figure(num="Questão 2 - Resposta ao impulso", figsize=(10, 4))
plt.plot(h, label="h[n] original")
plt.plot(np.arange(Nh), h_trunc, "--", label="h[n] truncada")
plt.title("Resposta ao impulso original e truncada")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 2.3 Filtragem por FFT no sinal completo
# ============================================================

print("\nQuestão 2 - Filtragem por FFT no áudio completo...")

y_fft = filtros_2.filtragemPorFFT(dados, h_trunc)
y_fft = y_fft[:len(dados)]

print("Filtragem por FFT concluída.")


# ============================================================
# 2.4 Convolução circular apenas em trecho pequeno
# ============================================================

print("\nQuestão 2 - Teste da convolução em trecho pequeno...")

N_teste = 3000

x_teste = dados[:N_teste]
y_conv_teste = filtros_2.filtragemPorConv(x_teste, h_trunc)
y_fft_teste = filtros_2.filtragemPorFFT(x_teste, h_trunc)

erro_conv_fft = np.max(np.abs(y_conv_teste - y_fft_teste))

print("Número de amostras no teste de convolução =", N_teste)
print("Erro máximo entre convolução circular e FFT no teste =", erro_conv_fft)


# ============================================================
# 2.5 Comparação gráfica no tempo
# ============================================================

tempo = np.arange(len(dados)) / fs

plt.figure(num="Questão 2 - Comparação dos métodos", figsize=(12, 6))

plt.plot(tempo, dados, label="Sinal corrompido", linewidth=0.5)
plt.plot(tempo, y_eqdif, label="Equação de diferenças", linewidth=0.8)
plt.plot(tempo, y_fft, "--", label="FFT com h[n] truncada", linewidth=0.8)

plt.title("Comparação dos métodos de filtragem")
plt.xlabel("t (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 2.6 Comparação numérica entre equação de diferenças e FFT
# ============================================================

erro_eqdif_fft = np.max(np.abs(y_eqdif - y_fft))

print("\nQuestão 2 - Comparação numérica:")
print("Erro máximo entre equação de diferenças e FFT =", erro_eqdif_fft)
print("Observação: a diferença ocorre porque a FFT usa h[n] truncada, enquanto a equação de diferenças implementa o IIR completo.")

#todo

#* Execução da terceira parte

#todo

#* Execução da quarta parte
