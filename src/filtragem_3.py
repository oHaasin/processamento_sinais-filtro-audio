import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import subprocess
from scipy.io import wavfile

import filtros_2


#! ============================================================
#! Funções auxiliares
#! ============================================================

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


#! ============================================================
#! 3.1 - Filtragem do sinal de áudio utilizando as três formas
#! ============================================================

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

    #! ------------------------------------------------------------
    #! 3.1a - Filtragem pela equação de diferenças
    #! ------------------------------------------------------------

    print("\nQuestão 3.1 - Filtragem pela equação de diferenças...")

    filtro_eqdif = filtros_2.FiltroEqDif(numerador, denominador)

    y_eqdif = np.zeros(Nx, dtype=np.float64)

    for n in range(Nx):
        y_eqdif[n] = filtro_eqdif.filtrar_amostra(dados[n])

    resultados["eqdif"] = y_eqdif

    print("Concluída.")
    print("Tamanho da saída por equação de diferenças =", len(y_eqdif))

    #! ------------------------------------------------------------
    #! 3.1b - Filtragem por convolução com h_trunc[n]
    #! ------------------------------------------------------------

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

    #! ------------------------------------------------------------
    #! 3.1c - Filtragem pela FFT
    #! ------------------------------------------------------------

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


#! ============================================================
#! 3.2 - Apresentação dos sinais no domínio do tempo
#! ============================================================

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


#! ============================================================
#! 3.2 - Apresentação dos sinais no domínio da frequência
#! ============================================================

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

    f_max_khz = (fs / 2) / 1000
    marcadores_x = np.arange(-20, 21, 5)

    for i, (nome, sinal) in enumerate(sinais):
        frequencias_khz, magnitude, fase = calcular_espectro(sinal, fs)

        # Evita log(0)
        magnitude = np.maximum(magnitude, 1e-12)

        # Magnitude em escala lin-log
        axs[i, 0].semilogy(frequencias_khz, magnitude, linewidth=0.5)
        axs[i, 0].set_title(f"{nome} - Magnitude")
        axs[i, 0].set_xlabel("f (kHz)")
        axs[i, 0].set_ylabel(r"$|Y(e^{j\omega})|$")
        axs[i, 0].grid(True, linestyle="--", alpha=0.6)

        axs[i, 0].set_xlim(-f_max_khz, f_max_khz)
        axs[i, 0].set_xticks(marcadores_x)
        axs[i, 0].set_ylim(1e-10, 1e-2)

        # Fase
        axs[i, 1].plot(frequencias_khz, fase, linewidth=0.3)
        axs[i, 1].set_title(f"{nome} - Fase")
        axs[i, 1].set_xlabel("f (kHz)")
        axs[i, 1].set_ylabel(r"$\theta(\omega)$")
        axs[i, 1].grid(True, linestyle="--", alpha=0.6)

        axs[i, 1].set_xlim(-f_max_khz, f_max_khz)
        axs[i, 1].set_xticks(marcadores_x)
        axs[i, 1].set_ylim(-np.pi, np.pi)

    plt.tight_layout()
    plt.show()
    
#! ============================================================
#! 3.3 - Execução do sinal filtrado no sistema de áudio
#! ============================================================


def converter_para_int16(sinal):
    """
    Converte um sinal em ponto flutuante para int16, formato comum de .wav.

    Se o sinal estiver em faixa [-1, 1], apenas escala para int16.
    Se passar dessa faixa, normaliza pelo valor máximo absoluto.
    """

    sinal = np.asarray(sinal, dtype=np.float64).flatten()

    max_abs = np.max(np.abs(sinal))

    if max_abs == 0:
        return np.zeros_like(sinal, dtype=np.int16)

    # Evita clipping caso a amplitude passe de [-1, 1]
    if max_abs > 1:
        sinal = sinal / max_abs

    sinal_int16 = np.int16(sinal * 32767)

    return sinal_int16


def salvar_audio_filtrado(fs, resultados, pasta_saida="audios_filtrados"):
    """
    Salva os sinais filtrados em arquivos .wav para posterior execução.

    Gera:
        audio_filtrado_eqdif.wav
        audio_filtrado_conv.wav
        audio_filtrado_fft.wav
    """

    os.makedirs(pasta_saida, exist_ok=True)

    caminhos = {}

    sinais = {
        "eqdif": resultados["eqdif"],
        "conv": resultados["conv"],
        "fft": resultados["fft"],
    }

    nomes_arquivos = {
        "eqdif": "audio_filtrado_eqdif.wav",
        "conv": "audio_filtrado_conv.wav",
        "fft": "audio_filtrado_fft.wav",
    }

    for metodo, sinal in sinais.items():
        sinal_int16 = converter_para_int16(sinal)

        caminho = os.path.join(pasta_saida, nomes_arquivos[metodo])


        wavfile.write(caminho, fs, sinal_int16)

        caminhos[metodo] = caminho

    print("\nQuestão 3.3 - Arquivos de áudio filtrado salvos:")
    for metodo, caminho in caminhos.items():
        print(f"{metodo}: {caminho}")

    return caminhos


def executar_audio(caminho_audio):
    """
    Executa um arquivo de áudio no player padrão do sistema operacional.
    """

    if sys.platform.startswith("win"):
        os.startfile(caminho_audio)

    elif sys.platform.startswith("darwin"):
        subprocess.run(["open", caminho_audio], check=False)

    else:
        subprocess.run(["xdg-open", caminho_audio], check=False)


def executar_audio_filtrado(fs, resultados, metodo="fft", pasta_saida="audios_filtrados"):
    """
    Salva os áudios filtrados e executa um deles no sistema de áudio.

    Parâmetros:
        fs: frequência de amostragem
        resultados: dicionário retornado por filtrar_sinal_tres_formas()
        metodo: "eqdif", "conv" ou "fft"
        pasta_saida: pasta onde os arquivos .wav serão salvos
    """

    caminhos = salvar_audio_filtrado(fs, resultados, pasta_saida=pasta_saida)

    if metodo not in caminhos:
        raise ValueError("Método inválido. Use: 'eqdif', 'conv' ou 'fft'.")

    print(f"\nExecutando áudio filtrado pelo método: {metodo}")
    executar_audio(caminhos[metodo])

    return caminhos


def executar_todos_audios_filtrados(fs, resultados, pasta_saida="audios_filtrados"):
    """
    Salva e executa os três sinais filtrados.
    """

    caminhos = salvar_audio_filtrado(fs, resultados, pasta_saida=pasta_saida)

    for metodo, caminho in caminhos.items():
        print(f"\nExecutando áudio filtrado pelo método: {metodo}")
        executar_audio(caminho)

    return caminhos


#! ============================================================
#! 3.4 - Análise dos efeitos da filtragem linear
#! ============================================================

def calcular_rms(sinal):
    """
    Calcula o valor RMS de um sinal.
    """

    sinal = np.asarray(sinal, dtype=np.float64).flatten()

    return np.sqrt(np.mean(sinal ** 2))


def calcular_energia_banda(sinal, fs, f_min, f_max):
    """
    Calcula a energia espectral aproximada em uma faixa de frequências.

    Usa FFT bilateral e considera |f| entre f_min e f_max.
    """

    sinal = np.asarray(sinal, dtype=np.float64).flatten()

    N = len(sinal)

    X = np.fft.fft(sinal)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    mascara = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)

    energia = np.sum(np.abs(X[mascara]) ** 2)

    return energia


def razao_db(valor_final, valor_inicial):
    """
    Calcula 10log10(valor_final / valor_inicial).
    """

    eps = 1e-20

    return 10 * np.log10((valor_final + eps) / (valor_inicial + eps))


def analisar_efeitos_filtragem(
    fs,
    sinal_original,
    resultados,
    faixa_ruido=(5000, 18000),
    faixa_util=(0, 5000),
    trecho_ruido=(16, 26)
):
    """
    Analisa os efeitos da filtragem linear sobre o sinal.

    Métricas calculadas:
        - RMS total;
        - RMS no trecho ruidoso;
        - energia na faixa útil;
        - energia na faixa de ruído;
        - atenuação em dB na faixa de ruído;
        - diferença entre métodos.

    Observação:
    Como não há sinal limpo de referência, a análise é feita comparando
    o sinal corrompido com os sinais filtrados.
    """

    sinal_original = np.asarray(sinal_original, dtype=np.float64).flatten()

    sinais = {
        "Equação de diferenças": resultados["eqdif"],
        "Convolução": resultados["conv"],
        "FFT": resultados["fft"],
    }

    f_ruido_min, f_ruido_max = faixa_ruido
    f_util_min, f_util_max = faixa_util

    idx_inicio_ruido = int(trecho_ruido[0] * fs)
    idx_fim_ruido = int(trecho_ruido[1] * fs)

    idx_inicio_ruido = max(0, idx_inicio_ruido)
    idx_fim_ruido = min(len(sinal_original), idx_fim_ruido)

    original_trecho_ruido = sinal_original[idx_inicio_ruido:idx_fim_ruido]

    rms_original = calcular_rms(sinal_original)
    rms_original_ruido = calcular_rms(original_trecho_ruido)

    energia_ruido_original = calcular_energia_banda(
        sinal_original,
        fs,
        f_ruido_min,
        f_ruido_max
    )

    energia_util_original = calcular_energia_banda(
        sinal_original,
        fs,
        f_util_min,
        f_util_max
    )

    print("\nQuestão 3.4 - Análise dos efeitos da filtragem linear")
    print("Faixa útil considerada:", faixa_util, "Hz")
    print("Faixa de ruído considerada:", faixa_ruido, "Hz")
    print("Trecho ruidoso considerado:", trecho_ruido, "s")

    print("\nSinal original corrompido:")
    print("RMS total =", rms_original)
    print("RMS no trecho ruidoso =", rms_original_ruido)
    print("Energia na faixa útil =", energia_util_original)
    print("Energia na faixa de ruído =", energia_ruido_original)

    metricas = {}

    for nome, sinal in sinais.items():
        sinal = np.asarray(sinal, dtype=np.float64).flatten()

        sinal_trecho_ruido = sinal[idx_inicio_ruido:idx_fim_ruido]

        rms_total = calcular_rms(sinal)
        rms_trecho_ruido = calcular_rms(sinal_trecho_ruido)

        energia_ruido = calcular_energia_banda(
            sinal,
            fs,
            f_ruido_min,
            f_ruido_max
        )

        energia_util = calcular_energia_banda(
            sinal,
            fs,
            f_util_min,
            f_util_max
        )

        atenuacao_ruido_db = razao_db(energia_ruido, energia_ruido_original)
        variacao_util_db = razao_db(energia_util, energia_util_original)

        metricas[nome] = {
            "rms_total": rms_total,
            "rms_trecho_ruido": rms_trecho_ruido,
            "energia_util": energia_util,
            "energia_ruido": energia_ruido,
            "atenuacao_ruido_db": atenuacao_ruido_db,
            "variacao_util_db": variacao_util_db,
        }

        print(f"\nMétodo: {nome}")
        print("RMS total =", rms_total)
        print("RMS no trecho ruidoso =", rms_trecho_ruido)
        print("Energia na faixa útil =", energia_util)
        print("Energia na faixa de ruído =", energia_ruido)
        print("Variação da energia útil em relação ao original (dB) =", variacao_util_db)
        print("Atenuação da energia na faixa de ruído em relação ao original (dB) =", atenuacao_ruido_db)

    # Diferenças entre métodos
    erro_eqdif_conv = np.max(np.abs(resultados["eqdif"] - resultados["conv"]))
    erro_eqdif_fft = np.max(np.abs(resultados["eqdif"] - resultados["fft"]))
    erro_conv_fft = np.max(np.abs(resultados["conv"] - resultados["fft"]))

    print("\nComparação entre os métodos:")
    print("Erro máximo entre equação de diferenças e convolução =", erro_eqdif_conv)
    print("Erro máximo entre equação de diferenças e FFT =", erro_eqdif_fft)
    print("Erro máximo entre convolução e FFT =", erro_conv_fft)

    print("\nAnálise qualitativa:")
    print("- A equação de diferenças implementa diretamente o filtro IIR completo.")
    print("- A convolução e a FFT usam a resposta ao impulso truncada, portanto podem apresentar diferenças em relação ao IIR completo.")
    print("- A convolução e a FFT tendem a produzir resultados praticamente iguais, pois representam a mesma operação linear quando usado zero-padding adequado.")
    print("- A filtragem reduz componentes na faixa de rejeição, mas pode atenuar altas frequências do áudio, causando sensação de som mais abafado.")
    print("- O método por FFT é mais adequado para blocos grandes do que a convolução direta implementada com laços, pois possui menor custo computacional.")

    return metricas


def plotar_metricas_filtragem(metricas):
    """
    Gera gráficos simples para comparar:
        - RMS total;
        - atenuação na faixa de ruído;
        - variação na faixa útil.
    """

    metodos = list(metricas.keys())

    rms_total = [metricas[m]["rms_total"] for m in metodos]
    atenuacao_ruido = [metricas[m]["atenuacao_ruido_db"] for m in metodos]
    variacao_util = [metricas[m]["variacao_util_db"] for m in metodos]

    plt.figure(num="Questão 3.4 - RMS total", figsize=(8, 4))
    plt.bar(metodos, rms_total)
    plt.title("RMS total dos sinais filtrados")
    plt.ylabel("RMS")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(num="Questão 3.4 - Atenuação da faixa de ruído", figsize=(8, 4))
    plt.bar(metodos, atenuacao_ruido)
    plt.title("Atenuação da energia na faixa de ruído")
    plt.ylabel("Atenuação em relação ao original (dB)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(num="Questão 3.4 - Variação da faixa útil", figsize=(8, 4))
    plt.bar(metodos, variacao_util)
    plt.title("Variação da energia na faixa útil")
    plt.ylabel("Variação em relação ao original (dB)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()