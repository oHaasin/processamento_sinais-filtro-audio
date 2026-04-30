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
# 2.1a - Filtragem pela equação de diferenças
# ============================================================
# A função é chamada amostra por amostra, simulando o comportamento
# de um conversor A/D. Para cada amostra x[n], retorna uma amostra y[n].

print("\nQuestão 2.1a - Filtragem por equação de diferenças...")

filtro_eqdif = filtros_2.FiltroEqDif(numerador, denominador)

y_eqdif = np.zeros_like(dados, dtype=np.float64)

for n in range(len(dados)):
    y_eqdif[n] = filtro_eqdif.filtrar_amostra(dados[n])

print("Filtragem por equação de diferenças concluída.")
print("Número de amostras processadas =", len(dados))
print("Número de amostras geradas =", len(y_eqdif))


# ============================================================
# 2.2a - Truncagem da resposta ao impulso
# ============================================================
# A resposta ao impulso h[n] é calculada com 1000 amostras.
# Em seguida, elimina-se a cauda cujas amostras são menores
# que 1% do valor de pico, mantendo as amostras iniciais.

print("\nQuestão 2.2a - Truncagem da resposta ao impulso...")

N_h_total = 1000

impulso = np.zeros(N_h_total)
impulso[0] = 1.0

filtro_impulso = filtros_2.FiltroEqDif(numerador, denominador)

h = np.zeros(N_h_total)

for n in range(N_h_total):
    h[n] = filtro_impulso.filtrar_amostra(impulso[n])

h_trunc, Nh = filtros_2.truncar_resposta_impulso(h, percentual=0.01)

print("Número de amostras original de h[n] =", len(h))
print("Pico de |h[n]| =", np.max(np.abs(h)))
print("Limiar de truncagem =", 0.01 * np.max(np.abs(h)))


# ============================================================
# 2.2b - Apresentação de h_trunc[n] e Nh
# ============================================================

print("\nQuestão 2.2b - Resposta ao impulso truncada:")
print("Número de amostras de h_trunc[n], Nh =", Nh)

plt.figure(num="Questão 2.2b - Resposta ao impulso truncada", figsize=(10, 4))
plt.plot(h, label="h[n] original")
plt.plot(np.arange(Nh), h_trunc, "--", label="h_trunc[n]")
plt.title("Resposta ao impulso original e truncada")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 2.2c - Filtragem por convolução circular
# ============================================================
# A função filtragemPorConv recebe um bloco x[n] de tamanho Nx
# e retorna y[n] com tamanho Ny = Nx + Nh - 1.
#
# Observação: para não travar o programa, a validação é feita
# em um trecho pequeno do áudio, pois a convolução circular direta
# é computacionalmente muito pesada para o áudio completo.

print("\nQuestão 2.2c - Validação da filtragem por convolução circular...")

N_teste = 3000
x_teste = dados[:N_teste]

y_conv_teste = filtros_2.filtragemPorConv(x_teste, h_trunc)

Nx_teste = len(x_teste)
Ny_esperado = Nx_teste + Nh - 1

print("Nx =", Nx_teste)
print("Nh =", Nh)
print("Ny esperado =", Ny_esperado)
print("Ny obtido pela convolução =", len(y_conv_teste))


# ============================================================
# 2.3a - Filtragem pela propriedade da multiplicação da FFT
# ============================================================
# Ainda usando h_trunc[n], a filtragem é feita por:
# y[n] = real{ifft(fft(x[n]) * fft(h_trunc[n]))}

print("\nQuestão 2.3a - Filtragem por FFT no áudio completo...")

y_fft_completo = filtros_2.filtragemPorFFT(dados, h_trunc)

print("Filtragem por FFT concluída.")


# ============================================================
# 2.3b - Validação do tamanho da saída da FFT
# ============================================================
# A função filtragemPorFFT deve receber um bloco de tamanho Nx
# e retornar y[n] com tamanho Ny = Nx + Nh - 1.

print("\nQuestão 2.3b - Validação da saída por FFT:")

Nx_audio = len(dados)
Ny_fft_esperado = Nx_audio + Nh - 1

print("Nx =", Nx_audio)
print("Nh =", Nh)
print("Ny esperado =", Ny_fft_esperado)
print("Ny obtido pela FFT =", len(y_fft_completo))

# Para comparação com o sinal original em gráficos futuros,
# corta-se a saída para o mesmo tamanho do áudio.
y_fft = y_fft_completo[:len(dados)]


# ============================================================
# Validação auxiliar - Convolução versus FFT no mesmo bloco
# ============================================================
# Esta validação não é um item separado do enunciado; serve apenas
# para confirmar numericamente que a implementação por FFT está
# equivalente à convolução circular com zero-padding.

print("\nValidação auxiliar - Convolução versus FFT:")

y_fft_teste = filtros_2.filtragemPorFFT(x_teste, h_trunc)

erro_conv_fft = np.max(np.abs(y_conv_teste - y_fft_teste))

print("Erro máximo entre convolução circular e FFT no teste =", erro_conv_fft)

# ============================================================
# Validação visual auxiliar - Comparação dos métodos
# ============================================================
# Este gráfico não é um item separado do enunciado da Questão 2.
# Ele serve para visualizar o resultado da equação de diferenças
# e da FFT com resposta ao impulso truncada.

tempo = np.arange(len(dados)) / fs

plt.figure(num="Questão 2 - Comparação dos métodos", figsize=(12, 6))

plt.plot(tempo, dados, label="Sinal corrompido", linewidth=0.5)
plt.plot(tempo, y_eqdif, label="Equação de diferenças", linewidth=0.8)
plt.plot(tempo, y_fft, "--", label="FFT com h[n] truncada", linewidth=0.8)

plt.title("Comparação dos métodos de filtragem")
plt.xlabel("t (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

#todo

#* Execução da terceira parte

# ============================================================
# 3. Filtragem do sinal
# ============================================================

resultados_3 = filtragem_3.filtrar_sinal_tres_formas(
    dados=dados,
    numerador=numerador,
    denominador=denominador,
    h_trunc=h_trunc,
    usar_convolucao_implementada_no_audio_completo=False,
    n_validacao_conv=3000
)

# 3.2 - Sinais no domínio do tempo
filtragem_3.plotar_sinais_filtrados_tempo(
    fs=fs,
    sinal_original=dados,
    resultados=resultados_3
)

# Zoom na região mais afetada pelo ruído
filtragem_3.plotar_sinais_filtrados_tempo(
    fs=fs,
    sinal_original=dados,
    resultados=resultados_3,
    t_inicio=16,
    t_fim=26
)

# 3.2 - Sinais no domínio da frequência
filtragem_3.plotar_espectros_filtrados(
    fs=fs,
    resultados=resultados_3
)

#todo

#* Execução da quarta parte
