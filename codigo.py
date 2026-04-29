"""
# Autores: Oliver Haas, Gabriel Paiva
# Data de última modificação: 28/04/2026
#? Objetivo: Filtrar ruído de um arquivo de som .wav
#todo: arrumar ampltidue errada, 1.5, 1.6, 2 inteiro, 3 inteiro, 4 bônus
"""

import os # Sistema operacional
import tkinter as tk # Interface
from tkinter import filedialog
import numpy as np # Calculos matemáticos
import matplotlib.pyplot as plt # Gráfico
from scipy.io import wavfile # Lê o arquivo .wav e extrai os dados numéricos do sinal

#! ==========================
#! 1.1
#! ==========================

# Esconde a janela principal do tkinter
root = tk.Tk()
root.withdraw()

#? Abre a janela para selecionar o arquivo (filtrando por .wav)
caminho_arquivo = filedialog.askopenfilename(
    title="Selecione o arquivo de áudio com o ruído",
    filetypes=[("Arquivos de Áudio", "*.wav")]
)

# Se o usuário não selecionou um arquivo ( cancelou a janela)
if not caminho_arquivo:
    raise FileNotFoundError("A operação foi cancelada: Nenhum arquivo de áudio foi selecionado.")

#? Abre o arquivo no programa padrão do Windows
os.startfile(caminho_arquivo)

#! ==========================
#! 1.2
#! ==========================

# Lê a taxa de amostragem (frequência) e o array de dados do sinal x[n]
taxa_amostragem, dados_sinal = wavfile.read(caminho_arquivo)

# Duração = número de amostras / taxa de amostragem (t = N/f)
#? linspace cria uma sequência de números uniformemente espaçados:
# 1 parâmetro: início (t = 0s)
# 2 parâmetro: final
# 3 parâmetro: quantidade de pontos
tempo = np.linspace(0, len(dados_sinal) / taxa_amostragem, num=len(dados_sinal))

# Configura e exibe o gráfico
plt.figure(num="Forma de onda em função do tempo", figsize=(10, 4)) # Nome da janela + 10 polegadas de largura, 4 de altura
plt.plot(tempo, dados_sinal, color='#1f77b4', linewidth=0.5) # Cor azul
plt.title("Sinal corrompido")
plt.xlabel("t(s)")
plt.ylabel("x(t)")
plt.grid(True, linestyle='--', alpha=0.6) # Ativa a grade de fundo, deixa as linhas tracejadas e 60% de opacidade
plt.xticks(np.arange(0, tempo[-1] + 5, 5)) # Define as marcações do eixo x começando em 0, até o tempo final, variando de 5 em 5
plt.xlim(0, tempo[-1]) # Garante que o eixo X comece em 0 e termine no fim do áudio
plt.tight_layout() # Comando automático de formatação

# Mostra a janela com o gráfico
plt.show()

#! ==========================
#! 1.3
#! ==========================

# Domínio da Frequência (FFT)
N = len(dados_sinal)

#? Aplica a Transformada Rápida de Fourier (FFT)
fft_resultado_bruto = np.fft.fft(dados_sinal)

#? Gera o eixo X das frequências originais (em Hz)
frequencias_hz_brutas = np.fft.fftfreq(N, d=1/taxa_amostragem)

#? o 'fftshift' centraliza a frequência zero (0 Hz)
# Corta os vetores pela metade (Teorema de Nyquist)
# A primeira metade (de 0 a f/2) contém as frequências positivas
# A segunda metade (de f/2 a f) contém as frequências negativas
fft_centralizada = np.fft.fftshift(fft_resultado_bruto)
frequencias_hz_centralizadas = np.fft.fftshift(frequencias_hz_brutas)

# Converte o eixo X para kHz
frequencias_khz = frequencias_hz_centralizadas / 1000

#? Calcula a Amplitude (Módulo) e Normaliza
# Quando usamos o espectro completo, normalizamos dividindo por N
amplitude = np.abs(fft_centralizada) / N

#? Calcula a Fase (em radianos)
fase = np.angle(fft_centralizada)


# Configura e exibe os gráficos
# Cria uma nova figura com dois subgráficos (2 linhas, 1 coluna)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), num="Espectro de Frequência")

# Define o limite exato de Nyquist
f_max_khz = (taxa_amostragem / 2) / 1000

# Cria uma lista de marcadores de 5 em 5. 
max_tick = int(f_max_khz // 5) * 5
marcadores_x = np.arange(-max_tick, max_tick + 1, 5)

# Subgráfico 1: Amplitude
ax1.plot(frequencias_khz, amplitude, color='#1f77b4', linewidth=0.5)
ax1.set_title("Espectro de Amplitude |X(e^jw)|")
ax1.set_xlabel("f (kHz)")
ax1.set_ylabel("Amplitude")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.set_xlim(-f_max_khz, f_max_khz) # Trava o eixo X para ir da frequência de Nyquist negativa até a positiva
ax1.set_xticks(marcadores_x)

# Subgráfico 2: Fase
ax2.plot(frequencias_khz, fase, color='#1f77b4', linewidth=0.1)
ax2.set_title("Espectro de Fase ∠X(e^jw)")
ax2.set_xlabel("f (kHz)")
ax2.set_ylabel("Fase (radianos)")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.set_xlim(-f_max_khz, f_max_khz)
ax2.set_xticks(marcadores_x)

plt.tight_layout()
plt.show()

#! ==========================
#! 1.4
#! ==========================

