import numpy as np
import matplotlib.pyplot as plt

import filtros_2


# ============================================================
# Funções auxiliares
# ============================================================

def ajustar_tamanho(y, tamanho):
    """
    Ajusta o sinal y para ter exatamente 'tamanho' amostras.
    Se y for maior, corta.
    Se y for menor, completa com zeros.
    """

    y = np.asarray(y, dtype=np.float64).flatten()

    if len(y) >= tamanho:
        return y[:tamanho]

    y_ajustado = np.zeros(tamanho, dtype=np.float64)
    y_ajustado[:len(y)] = y

    return y_ajustado


def calcular_espectro(sinal, fs):
    """
    Calcula o espectro centralizado de um sinal.

    Retorna:
        frequencias_khz: eixo de frequência em kHz
        magnitude: módulo do espectro
        fase: fase do espectro em radianos
    """

    sinal = np.asarray(sinal, dtype=np.float64).flatten()

    N = len(sinal)

    Y = np.fft.fftshift(np.fft.fft(sinal)) / N
    frequencias_hz = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))

    frequencias_khz = frequencias_hz / 1000
    magnitude = np.abs(Y)
    fase = np.angle(Y)

    return frequencias_khz, magnitude, fase


# ============================================================
# 3.1 - Filtragem do sinal de áudio utilizando as três formas
# ============================================================

def filtrar_sinal_tres_formas(
    dados,
    numerador,
    denominador,
    h_trunc,
    usar_convolucao_implementada_no_audio_completo=False,
    n_validacao_conv=3000
):
    """
    Filtra o sinal de áudio usando as três formas implementadas:

    1) equação de diferenças;
    2) convolução com h_trunc[n];
    3) propriedade da multiplicação da FFT.

    Observação importante:
    A função filtros_2.filtragemPorConv usa laços for em Python puro.
    Para o áudio completo, isso fica extremamente lento. Por isso, por
    padrão, a saída completa por convolução é calculada com np.convolve,
    que representa a mesma operação matemática de convolução linear.

    A função implementada no item 2.2c continua sendo usada para validação
    em um trecho pequeno do áudio.
    """

    dados = np.asarray(dados, dtype=np.float64).flatten()
    h_trunc = np.asarray(h_trunc, dtype=np.float64).flatten()

    Nx = len(dados)
    Nh = len(h_trunc)
    Ny = Nx + Nh - 1

    resultados = {
        "Nx": Nx,
        "Nh": Nh,
        "Ny": Ny,
    }

    # ------------------------------------------------------------
    # 3.1a - Filtragem pela equação de diferenças
    # ------------------------------------------------------------

    print("\nQuestão 3.1 - Filtragem pela equação de diferenças...")

    filtro_eqdif = filtros_2.FiltroEqDif(numerador, denominador)

    y_eqdif = np.zeros(Nx, dtype=np.float64)

    for n in range(Nx):
        y_eqdif[n] = filtro_eqdif.filtrar_amostra(dados[n])

    resultados["eqdif"] = y_eqdif

    print("Concluída.")
    print("Tamanho da saída por equação de diferenças =", len(y_eqdif))

    # ------------------------------------------------------------
    # 3.1b - Filtragem por convolução com h_trunc[n]
    # ------------------------------------------------------------

    print("\nQuestão 3.1 - Filtragem por convolução...")

    if usar_convolucao_implementada_no_audio_completo:
        print("Usando filtros_2.filtragemPorConv no áudio completo.")
        print("Aviso: essa opção pode demorar muito.")
        y_conv_completo = filtros_2.filtragemPorConv(dados, h_trunc)
    else:
        print("Usando np.convolve para obter a convolução completa do áudio.")
        print("A função implementada filtros_2.filtragemPorConv será validada em um trecho pequeno.")
        y_conv_completo = np.convolve(dados, h_trunc)

    y_conv = ajustar_tamanho(y_conv_completo, Nx)

    resultados["conv_completo"] = y_conv_completo
    resultados["conv"] = y_conv

    print("Concluída.")
    print("Ny esperado =", Ny)
    print("Tamanho da saída completa por convolução =", len(y_conv_completo))

    # Validação da função implementada no item 2.2c
    n_validacao_conv = min(n_validacao_conv, Nx)
    x_teste = dados[:n_validacao_conv]

    y_conv_teste = filtros_2.filtragemPorConv(x_teste, h_trunc)
    y_conv_ref = np.convolve(x_teste, h_trunc)

    erro_conv = np.max(np.abs(y_conv_teste - y_conv_ref))

    resultados["conv_validacao"] = y_conv_teste
    resultados["erro_conv_validacao"] = erro_conv

    print("\nValidação da função filtragemPorConv:")
    print("Nx teste =", len(x_teste))
    print("Nh =", Nh)
    print("Ny esperado no teste =", len(x_teste) + Nh - 1)
    print("Ny obtido no teste =", len(y_conv_teste))
    print("Erro máximo em relação ao np.convolve =", erro_conv)

    # ------------------------------------------------------------
    # 3.1c - Filtragem pela FFT
    # ------------------------------------------------------------

    print("\nQuestão 3.1 - Filtragem por FFT...")

    y_fft_completo = filtros_2.filtragemPorFFT(dados, h_trunc)
    y_fft = ajustar_tamanho(y_fft_completo, Nx)

    resultados["fft_completo"] = y_fft_completo
    resultados["fft"] = y_fft

    print("Concluída.")
    print("Ny esperado =", Ny)
    print("Tamanho da saída completa por FFT =", len(y_fft_completo))

    # Validação entre convolução e FFT
    erro_conv_fft = np.max(np.abs(y_conv_completo - y_fft_completo))

    resultados["erro_conv_fft"] = erro_conv_fft

    print("\nValidação entre convolução e FFT:")
    print("Erro máximo entre convolução e FFT =", erro_conv_fft)

    return resultados


# ============================================================
# 3.2 - Apresentação dos sinais no domínio do tempo
# ============================================================

def plotar_sinais_filtrados_tempo(
    fs,
    sinal_original,
    resultados,
    t_inicio=None,
    t_fim=None
):
    """
    Plota o sinal original e os sinais filtrados no domínio do tempo.

    Se t_inicio e t_fim forem fornecidos, faz zoom nesse intervalo.
    """

    sinal_original = np.asarray(sinal_original, dtype=np.float64).flatten()

    y_eqdif = resultados["eqdif"]
    y_conv = resultados["conv"]
    y_fft = resultados["fft"]

    N = len(sinal_original)
    tempo = np.arange(N) / fs

    if t_inicio is not None and t_fim is not None:
        idx_inicio = int(t_inicio * fs)
        idx_fim = int(t_fim * fs)

        idx_inicio = max(0, idx_inicio)
        idx_fim = min(N, idx_fim)

        tempo_plot = tempo[idx_inicio:idx_fim]
        sinal_plot = sinal_original[idx_inicio:idx_fim]
        eqdif_plot = y_eqdif[idx_inicio:idx_fim]
        conv_plot = y_conv[idx_inicio:idx_fim]
        fft_plot = y_fft[idx_inicio:idx_fim]

        titulo = f"Sinais filtrados no tempo - zoom de {t_inicio}s a {t_fim}s"
        nome_janela = "Questão 3.2 - Tempo - Zoom"
    else:
        tempo_plot = tempo
        sinal_plot = sinal_original
        eqdif_plot = y_eqdif
        conv_plot = y_conv
        fft_plot = y_fft

        titulo = "Sinais filtrados no domínio do tempo"
        nome_janela = "Questão 3.2 - Tempo"

    plt.figure(num=nome_janela, figsize=(12, 6))

    plt.plot(tempo_plot, sinal_plot, label="Sinal corrompido", linewidth=0.5)
    plt.plot(tempo_plot, eqdif_plot, label="Equação de diferenças", linewidth=0.8)
    plt.plot(tempo_plot, conv_plot, "--", label="Convolução com h_trunc[n]", linewidth=0.8)
    plt.plot(tempo_plot, fft_plot, ":", label="FFT com h_trunc[n]", linewidth=0.8)

    plt.title(titulo)
    plt.xlabel("t (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# 3.2 - Apresentação dos sinais no domínio da frequência
# ============================================================

def plotar_espectros_filtrados(fs, resultados):
    """
    Plota os espectros de amplitude e fase dos sinais filtrados
    pelos três métodos.

    A magnitude é apresentada em escala lin-log:
    eixo x linear em frequência e eixo y logarítmico em magnitude.
    """

    sinais = [
        ("Equação de diferenças", resultados["eqdif"]),
        ("Convolução", resultados["conv"]),
        ("FFT", resultados["fft"]),
    ]

    fig, axs = plt.subplots(
        3,
        2,
        figsize=(14, 10),
        num="Questão 3.2 - Espectros dos sinais filtrados"
    )

    for i, (nome, sinal) in enumerate(sinais):
        frequencias_khz, magnitude, fase = calcular_espectro(sinal, fs)

        magnitude = magnitude + 1e-12

        # Magnitude em escala lin-log
        axs[i, 0].semilogy(frequencias_khz, magnitude, linewidth=0.5)
        axs[i, 0].set_title(f"{nome} - Magnitude")
        axs[i, 0].set_xlabel("f (kHz)")
        axs[i, 0].set_ylabel("|Y(e^jω)|")
        axs[i, 0].grid(True, linestyle="--", alpha=0.6)

        # Fase
        axs[i, 1].plot(frequencias_khz, fase, linewidth=0.3)
        axs[i, 1].set_title(f"{nome} - Fase")
        axs[i, 1].set_xlabel("f (kHz)")
        axs[i, 1].set_ylabel("θ(ω)")
        axs[i, 1].grid(True, linestyle="--", alpha=0.6)

        f_max_khz = (fs / 2) / 1000

        axs[i, 0].set_xlim(-f_max_khz, f_max_khz)
        axs[i, 1].set_xlim(-f_max_khz, f_max_khz)

    plt.tight_layout()
    plt.show()