"""
# Data de última modificação: 29/04/2026
#? Objetivo: Filtrar ruído de um arquivo de som .wav
#todo: arrumar amplitude errada, 1.6
"""

import os # Sistema operacional
import tkinter as tk # Interface
from tkinter import filedialog
import numpy as np # Calculos matemáticos
import matplotlib.pyplot as plt # Gráfico
from scipy.io import wavfile # Lê o arquivo .wav e extrai os dados numéricos do sinal
import scipy.io as sio
import scipy.signal as signal

#! ==========================
#! 1.1 - Carregamento do .wav
#! ==========================

def carregar_wav():
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
    
    return caminho_arquivo

#! =========================
#! 1.2 Gráfico x(t) no tempo
#! =========================

def plotar_sinal_tempo(caminho_arquivo):
    # Lê a taxa de amostragem (frequência) e o array de dados do sinal x[n]
    frequencia_amostragem, dados_sinal = wavfile.read(caminho_arquivo)

    # Transforma os valores de amostra de -32768 até 32767 para -1 a 1.
    dados_sinal = dados_sinal.astype(np.float64) / 32768

    # Cria o eixo de tempo em segundos.
    #? np.arange(len(dados_sinal)) gera os índices das amostras: 0, 1, 2, ..., N-1.
    # Dividindo pela frequência de amostragem fs, cada índice n vira o instante t = n/fs.
    tempo = np.arange(len(dados_sinal)) / frequencia_amostragem

    # Configura e exibe o gráfico
    plt.figure(num="Forma de onda em função do tempo", figsize=(10, 4)) # Nome da janela + 10 polegadas de largura, 4 de altura
    plt.plot(tempo, dados_sinal, color='#1f77b4', linewidth=0.5) # Cor azul
    plt.title("Sinal corrompido")
    plt.xlabel("t(s)")
    plt.ylabel("x(t)")
    plt.grid(True, linestyle='--', alpha=0.6) # Ativa a grade de fundo, deixa as linhas tracejadas e 60% de opacidade
    plt.xticks(np.arange(0, tempo[-1] + 5, 5)) # Define as marcações do eixo x começando em 0, até o tempo final, variando de 5 em 5
    plt.xlim(0, tempo[-1]) # Garante que o eixo X comece em 0 e termine no fim do áudio
    plt.yticks(np.arange(-1, 1.01, 0.5))
    plt.tight_layout() # Comando automático de formatação

    # Mostra a janela com o gráfico
    plt.show()
    
    return frequencia_amostragem, dados_sinal

#! ==================================
#! 1.3 Gráficos X(e^jw) na frequência
#! ==================================

def plotar_espectro_frequencia(frequencia_amostragem, dados_sinal):
    
    # N = f_s * T -> número total de amostras
    N = len(dados_sinal)

    #? Aplica a Transformada Rápida de Fourier (FFT)
    # DFT direta: mais lenta, custo aproximado N^2
    # FFT: muito mais rápida, custo aproximado Nlog2N
    fft_resultado_bruto = np.fft.fft(dados_sinal)

    #? Gera o eixo X das frequências originais (em Hz)
    # A função fftfreq retorna as frequências associadas aos índices da FFT.
    frequencias_hz_brutas = np.fft.fftfreq(N, 1/frequencia_amostragem)

    #? o 'fftshift' centraliza a frequência zero (0 Hz)
    # Reorganiza os dados em duas partes:
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


    #* Configura e exibe os gráficos
    # Cria uma nova figura com dois subgráficos (2 linhas, 1 coluna)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), num="Espectro da entrada na frequência")

    # Define o limite exato de Nyquist
    f_max_khz = (frequencia_amostragem / 2) / 1000

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

#! =========================
#! 1.4 Carregamento dos .mat
#! =========================

def carrega_mat():

    caminho_num = filedialog.askopenfilename(
        title="Selecione o arquivo .mat do NUMERADOR",
        filetypes=[("Arquivos MATLAB", "*.mat")]
    )

    if not caminho_num:
        raise FileNotFoundError("Operação cancelada: Arquivo do numerador não selecionado.")

    caminho_den = filedialog.askopenfilename(
        title="Selecione o arquivo .mat do DENOMINADOR",
        filetypes=[("Arquivos MATLAB", "*.mat")]
    )

    if not caminho_den:
        raise FileNotFoundError("Operação cancelada: Arquivo do denominador não selecionado.")

    # Leitura e Extração Dinâmica dos Coeficientes
    mat_num = sio.loadmat(caminho_num)
    mat_den = sio.loadmat(caminho_den)

    # Adquire a primeira variável válida do arquivo
    def extrair_coeficientes(dicionario_mat):
        for chave, valor in dicionario_mat.items():
                if not chave.startswith('__'):  # Ignora os metadados
                    return valor.flatten()      # Transforma em array
        raise ValueError("Nenhuma variável de coeficiente encontrada no arquivo .mat")

    # Extrai os coeficientes sem precisar saber os nomes das variáveis
    numerador = extrair_coeficientes(mat_num)
    denominador = extrair_coeficientes(mat_den)

    return numerador, denominador

#! ==================================
#! 1.5 Gráficos H(e^jw) na frequência
#! ==================================

def plotar_resposta_frequencia(num, den, frequencia_amostragem):
    #? O filtro IIR é descrito por uma função de transferência do tipo:
    # H(e^jw) = B(e^jw) / A(e^jw)

    #* Espectro Completo (-fs/2 até fs/2), para escala linear
    # whole=True calcula o círculo unitário inteiro (0 até fs)
    # worN=8192 significa que o gráfico será calculado com 8192 pontos de frequência, não que o filtro tem 8192 pontos
    freqs_full_hz, h_full = signal.freqz(num, den, worN=8192, whole=True, fs=frequencia_amostragem)

    # Mover as frequências > fs/2 para a faixa negativa
    freqs_full_hz = freqs_full_hz - frequencia_amostragem * (freqs_full_hz > frequencia_amostragem / 2)

    # Reordena do menor (negativo) para o maior (positivo)
    idx_sort = np.argsort(freqs_full_hz)
    freqs_full_khz = freqs_full_hz[idx_sort] / 1000
    h_full_ordenado = h_full[idx_sort]

    magnitude_linear = np.abs(h_full_ordenado)
    fase_linear_rad = np.unwrap(np.angle(h_full_ordenado))
    # A fase é definida módulo 2π. Este deslocamento é apenas visual
    fase_linear_rad = fase_linear_rad - 5 * 2 * np.pi

    #* Apenas Frequências Positivas (0 até fs/2), para escala em dB
    freqs_pos_hz, h_pos = signal.freqz(num, den, worN=8192, fs=frequencia_amostragem)
    freqs_pos_khz = freqs_pos_hz / 1000

    #? 1e-20 evita log10(0), que é indefinido.
    magnitude_db = 20 * np.log10(np.abs(h_pos) + 1e-20)
    fase_graus = np.unwrap(np.angle(h_pos)) * (180 / np.pi)

    # Plot do painel 2x2
    fig, axs = plt.subplots(2, 2, figsize=(16, 8), num="Resposta ao impulso na frequência")

    # Ajuste do limite X
    f_max = (frequencia_amostragem / 2) / 1000

    # ---------------------------------------------------------
    # GRÁFICO 1 (Topo Esquerda): Magnitude Linear
    # ---------------------------------------------------------
    axs[0, 0].plot(freqs_full_khz, magnitude_linear, color='#1f77b4', linewidth=1)
    axs[0, 0].set_title("Magnitude em escala linear", fontsize=10, fontweight='bold')
    axs[0, 0].set_ylabel("|H(e^jw)|")
    axs[0, 0].set_xlabel("Frequência (kHz)")
    axs[0, 0].grid(True, linestyle='-', alpha=0.3)
    axs[0, 0].set_xlim(-f_max, f_max)

    # ---------------------------------------------------------
    # GRÁFICO 2 (Base Esquerda): Fase Linear (Radianos)
    # ---------------------------------------------------------
    axs[1, 0].plot(freqs_full_khz, fase_linear_rad, color='#1f77b4', linewidth=1)
    axs[1, 0].set_title("Fase em escala linear (Radianos)", fontsize=10, fontweight='bold')
    axs[1, 0].set_ylabel("θ(ω)")
    axs[1, 0].set_xlabel("Frequência (kHz)")
    axs[1, 0].grid(True, linestyle='-', alpha=0.3)
    axs[1, 0].set_xlim(-f_max, f_max)

    # ---------------------------------------------------------
    # GRÁFICO 3 (Topo Direita): Magnitude em dB (Lin-Log)
    # ---------------------------------------------------------
    axs[0, 1].plot(freqs_pos_khz, magnitude_db, color='#1f77b4', linewidth=1)
    axs[0, 1].set_title("Magnitude em escala lin-log", fontsize=10, fontweight='bold')
    axs[0, 1].set_ylabel("Magnitude (dB)")
    axs[0, 1].set_xlabel("Frequência (kHz)")
    axs[0, 1].grid(True, linestyle='-', alpha=0.3)
    axs[0, 1].set_xlim(0, f_max)
    axs[0, 1].set_ylim(-400, 20) # Força o limite Y similar ao da sua imagem

    # ---------------------------------------------------------
    # GRÁFICO 4 (Base Direita): Fase em Graus
    # ---------------------------------------------------------
    axs[1, 1].plot(freqs_pos_khz, fase_graus, color='#1f77b4', linewidth=1)
    axs[1, 1].set_title("Fase em escala lin-log", fontsize=10, fontweight='bold')
    axs[1, 1].set_ylabel("Fase (graus)")
    axs[1, 1].set_xlabel("Frequência (kHz)")
    axs[1, 1].grid(True, linestyle='-', alpha=0.3)
    axs[1, 1].set_xlim(0, f_max)

    # Ajusta o espaçamento entre os gráficos para nada ficar sobreposto
    plt.tight_layout()
    plt.show()

#! ==================================
#! 1.6 Gráfico h[n] por 1000 amostras
#! ==================================

#todo