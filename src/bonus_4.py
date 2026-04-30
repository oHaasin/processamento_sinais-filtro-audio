import time
import numpy as np
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import filtros_2


# ============================================================
# Função auxiliar
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


# ============================================================
# 4.1 - Algoritmo overlap-add genérico
# ============================================================

def overlap_add(x, h_trunc, tamanho_bloco=None, metodo="fft"):
    """
    Implementa a filtragem por blocos usando o método overlap-add.

    A ideia é dividir o sinal x[n] em blocos menores, filtrar cada
    bloco separadamente e somar as partes sobrepostas na saída.

    Parâmetros:
        x: sinal de entrada
        h_trunc: resposta ao impulso truncada
        tamanho_bloco: tamanho de cada bloco de entrada
        metodo:
            "conv" -> usa filtros_2.filtragemPorConv
            "fft"  -> usa filtros_2.filtragemPorFFT

    Retorna:
        y: sinal filtrado completo, de tamanho len(x) + len(h_trunc) - 1
    """

    x = np.asarray(x, dtype=np.float64).flatten()
    h_trunc = np.asarray(h_trunc, dtype=np.float64).flatten()

    Nx_total = len(x)
    Nh = len(h_trunc)

    if tamanho_bloco is None:
        tamanho_bloco = Nh

    if tamanho_bloco <= 0:
        raise ValueError("O tamanho do bloco deve ser maior que zero.")

    Ny_total = Nx_total + Nh - 1

    y = np.zeros(Ny_total, dtype=np.float64)

    for inicio in range(0, Nx_total, tamanho_bloco):
        fim = min(inicio + tamanho_bloco, Nx_total)

        bloco = x[inicio:fim]

        if metodo == "conv":
            y_bloco = filtros_2.filtragemPorConv(bloco, h_trunc)

        elif metodo == "fft":
            y_bloco = filtros_2.filtragemPorFFT(bloco, h_trunc)

        else:
            raise ValueError("Método inválido. Use 'conv' ou 'fft'.")

        inicio_saida = inicio
        fim_saida = inicio_saida + len(y_bloco)

        y[inicio_saida:fim_saida] += y_bloco

    return y


# ============================================================
# 4.1 - Versões específicas do overlap-add
# ============================================================

def overlap_add_conv(x, h_trunc, tamanho_bloco=None):
    """
    Overlap-add usando a função filtragemPorConv desenvolvida na Questão 2.
    """

    return overlap_add(
        x=x,
        h_trunc=h_trunc,
        tamanho_bloco=tamanho_bloco,
        metodo="conv"
    )


def overlap_add_fft(x, h_trunc, tamanho_bloco=None):
    """
    Overlap-add usando a função filtragemPorFFT desenvolvida na Questão 2.
    """

    return overlap_add(
        x=x,
        h_trunc=h_trunc,
        tamanho_bloco=tamanho_bloco,
        metodo="fft"
    )


# ============================================================
# 4.2 - Filtragem do áudio com blocos Nx = Nh
# ============================================================

def filtrar_audio_overlap_add(dados, h_trunc, metodos=("fft",), retornar_mesmo_tamanho=True):
    """
    Filtra o áudio usando overlap-add com blocos de tamanho Nx = Nh.

    Parâmetros:
        dados: sinal de áudio de entrada
        h_trunc: resposta ao impulso truncada
        metodos:
            ("fft",)         -> executa apenas overlap-add por FFT
            ("conv",)        -> executa apenas overlap-add por convolução
            ("conv", "fft")  -> executa os dois métodos
        retornar_mesmo_tamanho:
            True  -> também retorna as saídas cortadas para o tamanho do áudio
            False -> retorna apenas as saídas completas

    Retorna:
        resultados: dicionário com os sinais filtrados e informações da execução
    """

    dados = np.asarray(dados, dtype=np.float64).flatten()
    h_trunc = np.asarray(h_trunc, dtype=np.float64).flatten()

    Nx_total = len(dados)
    Nh = len(h_trunc)
    tamanho_bloco = Nh
    Ny_total = Nx_total + Nh - 1

    resultados = {
        "Nx_total": Nx_total,
        "Nh": Nh,
        "tamanho_bloco": tamanho_bloco,
        "Ny_total": Ny_total,
    }

    print("\nQuestão 4.2 - Filtragem por overlap-add")
    print("Tamanho total do sinal de entrada =", Nx_total)
    print("Tamanho da resposta ao impulso truncada Nh =", Nh)
    print("Tamanho dos blocos Nx =", tamanho_bloco)
    print("Tamanho esperado da saída Ny =", Ny_total)

    if "conv" in metodos:
        print("\nExecutando overlap-add por convolução...")
        print("Aviso: este método pode ser lento, pois usa convolução circular com laços.")

        t0 = time.perf_counter()
        y_ola_conv_completo = overlap_add_conv(
            x=dados,
            h_trunc=h_trunc,
            tamanho_bloco=tamanho_bloco
        )
        t1 = time.perf_counter()

        resultados["conv_completo"] = y_ola_conv_completo
        resultados["tempo_conv"] = t1 - t0

        if retornar_mesmo_tamanho:
            resultados["conv"] = ajustar_tamanho(y_ola_conv_completo, Nx_total)

        print("Overlap-add por convolução concluído.")
        print("Tempo de execução =", resultados["tempo_conv"], "s")
        print("Tamanho obtido =", len(y_ola_conv_completo))

    if "fft" in metodos:
        print("\nExecutando overlap-add por FFT...")

        t0 = time.perf_counter()
        y_ola_fft_completo = overlap_add_fft(
            x=dados,
            h_trunc=h_trunc,
            tamanho_bloco=tamanho_bloco
        )
        t1 = time.perf_counter()

        resultados["fft_completo"] = y_ola_fft_completo
        resultados["tempo_fft"] = t1 - t0

        if retornar_mesmo_tamanho:
            resultados["fft"] = ajustar_tamanho(y_ola_fft_completo, Nx_total)

        print("Overlap-add por FFT concluído.")
        print("Tempo de execução =", resultados["tempo_fft"], "s")
        print("Tamanho obtido =", len(y_ola_fft_completo))

    if "conv" in metodos and "fft" in metodos:
        erro_conv_fft = np.max(
            np.abs(resultados["conv_completo"] - resultados["fft_completo"])
        )

        resultados["erro_conv_fft"] = erro_conv_fft

        print("\nComparação overlap-add:")
        print("Erro máximo entre overlap-add por convolução e por FFT =", erro_conv_fft)

    return resultados


# ============================================================
# Validação auxiliar em trecho pequeno
# ============================================================

def validar_overlap_add(dados, h_trunc, n_validacao=3000):
    """
    Valida o overlap-add em um trecho pequeno do sinal.

    Compara:
        - overlap-add por convolução;
        - overlap-add por FFT;
        - convolução direta por np.convolve.

    Essa função é útil porque a versão por convolução implementada
    com laços pode ser lenta no áudio completo.
    """

    dados = np.asarray(dados, dtype=np.float64).flatten()
    h_trunc = np.asarray(h_trunc, dtype=np.float64).flatten()

    n_validacao = min(n_validacao, len(dados))

    x_teste = dados[:n_validacao]

    Nh = len(h_trunc)
    tamanho_bloco = Nh

    print("\nValidação auxiliar da Questão 4.1")
    print("Tamanho do trecho de teste =", len(x_teste))
    print("Nh =", Nh)
    print("Tamanho dos blocos Nx =", tamanho_bloco)

    y_ref = np.convolve(x_teste, h_trunc)

    y_ola_conv = overlap_add_conv(
        x=x_teste,
        h_trunc=h_trunc,
        tamanho_bloco=tamanho_bloco
    )

    y_ola_fft = overlap_add_fft(
        x=x_teste,
        h_trunc=h_trunc,
        tamanho_bloco=tamanho_bloco
    )

    erro_conv_ref = np.max(np.abs(y_ola_conv - y_ref))
    erro_fft_ref = np.max(np.abs(y_ola_fft - y_ref))
    erro_conv_fft = np.max(np.abs(y_ola_conv - y_ola_fft))

    print("Erro máximo entre overlap-add conv e np.convolve =", erro_conv_ref)
    print("Erro máximo entre overlap-add FFT e np.convolve =", erro_fft_ref)
    print("Erro máximo entre overlap-add conv e overlap-add FFT =", erro_conv_fft)

    resultados_validacao = {
        "x_teste": x_teste,
        "y_ref": y_ref,
        "y_ola_conv": y_ola_conv,
        "y_ola_fft": y_ola_fft,
        "erro_conv_ref": erro_conv_ref,
        "erro_fft_ref": erro_fft_ref,
        "erro_conv_fft": erro_conv_fft,
    }

    return resultados_validacao

# ============================================================
# Funções auxiliares para 4.3 e 4.4
# ============================================================

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


def obter_sinais_overlap_add(resultados_4):
    """
    Retorna os sinais disponíveis no dicionário resultados_4.

    O dicionário pode ter apenas FFT ou pode ter convolução e FFT,
    dependendo de como filtrar_audio_overlap_add foi chamada.
    """

    sinais = []

    if "conv" in resultados_4:
        sinais.append(("Overlap-add por convolução", resultados_4["conv"]))

    if "fft" in resultados_4:
        sinais.append(("Overlap-add por FFT", resultados_4["fft"]))

    if not sinais:
        raise ValueError("Nenhum sinal filtrado encontrado em resultados_4.")

    return sinais


# ============================================================
# 4.3 - Apresentação dos resultados no domínio do tempo
# ============================================================

def plotar_overlap_add_tempo(fs, sinal_original, resultados_4, t_inicio=None, t_fim=None):
    """
    Plota os sinais filtrados por overlap-add no domínio do tempo.

    Se t_inicio e t_fim forem fornecidos, faz zoom nesse intervalo.
    """

    sinal_original = np.asarray(sinal_original, dtype=np.float64).flatten()

    sinais = obter_sinais_overlap_add(resultados_4)

    N = len(sinal_original)
    tempo = np.arange(N) / fs

    if t_inicio is not None and t_fim is not None:
        idx_inicio = int(t_inicio * fs)
        idx_fim = int(t_fim * fs)

        idx_inicio = max(0, idx_inicio)
        idx_fim = min(N, idx_fim)

        tempo_plot = tempo[idx_inicio:idx_fim]
        sinal_original_plot = sinal_original[idx_inicio:idx_fim]

        titulo = f"Overlap-add no domínio do tempo - zoom de {t_inicio}s a {t_fim}s"
        nome_janela = "Questão 4.3 - Tempo - Zoom"
    else:
        idx_inicio = 0
        idx_fim = N

        tempo_plot = tempo
        sinal_original_plot = sinal_original

        titulo = "Overlap-add no domínio do tempo"
        nome_janela = "Questão 4.3 - Tempo"

    plt.figure(num=nome_janela, figsize=(12, 6))

    plt.plot(
        tempo_plot,
        sinal_original_plot,
        label="Sinal corrompido",
        linewidth=0.5
    )

    for nome, sinal in sinais:
        sinal = ajustar_tamanho(sinal, N)
        sinal_plot = sinal[idx_inicio:idx_fim]

        plt.plot(
            tempo_plot,
            sinal_plot,
            label=nome,
            linewidth=0.8
        )

    plt.title(titulo)
    plt.xlabel("t (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4.3 - Apresentação dos resultados no domínio da frequência
# ============================================================

def plotar_overlap_add_frequencia(fs, resultados_4):
    """
    Plota magnitude e fase dos sinais filtrados por overlap-add.

    A magnitude é apresentada em escala lin-log:
    eixo x linear em frequência e eixo y logarítmico.
    """

    sinais = obter_sinais_overlap_add(resultados_4)

    fig, axs = plt.subplots(
        len(sinais),
        2,
        figsize=(14, 4 * len(sinais)),
        num="Questão 4.3 - Espectros overlap-add"
    )

    if len(sinais) == 1:
        axs = np.array([axs])

    f_max_khz = (fs / 2) / 1000
    marcadores_x = np.arange(-20, 21, 5)

    for i, (nome, sinal) in enumerate(sinais):
        frequencias_khz, magnitude, fase = calcular_espectro(sinal, fs)

        magnitude = np.maximum(magnitude, 1e-12)

        # Magnitude
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


# ============================================================
# 4.4 - Execução do sinal filtrado no sistema de áudio
# ============================================================

def converter_para_int16(sinal):
    """
    Converte um sinal em ponto flutuante para int16, formato comum de .wav.
    """

    sinal = np.asarray(sinal, dtype=np.float64).flatten()

    max_abs = np.max(np.abs(sinal))

    if max_abs == 0:
        return np.zeros_like(sinal, dtype=np.int16)

    if max_abs > 1:
        sinal = sinal / max_abs

    return np.int16(sinal * 32767)


def salvar_audio_overlap_add(fs, resultados_4, pasta_saida="audios_filtrados"):
    """
    Salva os sinais filtrados por overlap-add em arquivos .wav.
    """

    os.makedirs(pasta_saida, exist_ok=True)

    caminhos = {}

    if "conv" in resultados_4:
        caminho_conv = os.path.join(pasta_saida, "audio_overlap_add_conv.wav")
        wavfile.write(caminho_conv, fs, converter_para_int16(resultados_4["conv"]))
        caminhos["conv"] = caminho_conv

    if "fft" in resultados_4:
        caminho_fft = os.path.join(pasta_saida, "audio_overlap_add_fft.wav")
        wavfile.write(caminho_fft, fs, converter_para_int16(resultados_4["fft"]))
        caminhos["fft"] = caminho_fft

    print("\nQuestão 4.4 - Arquivos de áudio overlap-add salvos:")
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


def executar_audio_overlap_add(fs, resultados_4, metodo="fft", pasta_saida="audios_filtrados"):
    """
    Salva e executa o áudio filtrado por overlap-add.

    metodo:
        "fft"  -> executa o áudio gerado por overlap-add FFT
        "conv" -> executa o áudio gerado por overlap-add convolução
    """

    caminhos = salvar_audio_overlap_add(
        fs=fs,
        resultados_4=resultados_4,
        pasta_saida=pasta_saida
    )

    if metodo not in caminhos:
        raise ValueError("Método não disponível. Use 'fft' ou 'conv', desde que tenha sido calculado.")

    print(f"\nExecutando áudio overlap-add pelo método: {metodo}")
    executar_audio(caminhos[metodo])

    return caminhos

# ============================================================
# 4.5 - Análise comparativa com os métodos anteriores
# ============================================================

def calcular_rms(sinal):
    """
    Calcula o valor RMS de um sinal.
    """

    sinal = np.asarray(sinal, dtype=np.float64).flatten()
    return np.sqrt(np.mean(sinal ** 2))


def calcular_energia_banda(sinal, fs, f_min, f_max):
    """
    Calcula a energia espectral aproximada em uma faixa de frequências.

    Usa a FFT bilateral e considera as frequências com:
        f_min <= |f| <= f_max
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


def analisar_comparativo_overlap_add(
    fs,
    sinal_original,
    resultados_3,
    resultados_4,
    validacao_4=None,
    faixa_ruido=(5000, 18000),
    faixa_util=(0, 5000),
    trecho_ruido=(16, 26)
):
    """
    Questão 4.5 - Análise comparativa entre:

        - métodos anteriores da Questão 3;
        - método overlap-add da Questão 4.

    Analisa:
        - qualidade do resultado;
        - eficiência na eliminação do ruído;
        - efeitos da truncagem da resposta ao impulso;
        - distorções resultantes;
        - desempenho computacional.
    """

    sinal_original = np.asarray(sinal_original, dtype=np.float64).flatten()
    N = len(sinal_original)

    f_util_min, f_util_max = faixa_util
    f_ruido_min, f_ruido_max = faixa_ruido

    idx_inicio_ruido = int(trecho_ruido[0] * fs)
    idx_fim_ruido = int(trecho_ruido[1] * fs)

    idx_inicio_ruido = max(0, idx_inicio_ruido)
    idx_fim_ruido = min(N, idx_fim_ruido)

    # ------------------------------------------------------------
    # Lista de sinais para comparação
    # ------------------------------------------------------------

    sinais = {
        "Original corrompido": sinal_original,
        "Eq. diferenças": resultados_3["eqdif"],
        "Convolução": resultados_3["conv"],
        "FFT": resultados_3["fft"],
    }

    if "conv" in resultados_4:
        sinais["Overlap-add conv"] = resultados_4["conv"]

    if "fft" in resultados_4:
        sinais["Overlap-add FFT"] = resultados_4["fft"]

    # Ajusta todos os sinais para o mesmo tamanho
    for nome in sinais:
        sinais[nome] = ajustar_tamanho(sinais[nome], N)

    # ------------------------------------------------------------
    # Métricas do sinal original
    # ------------------------------------------------------------

    energia_util_original = calcular_energia_banda(
        sinal_original,
        fs,
        f_util_min,
        f_util_max
    )

    energia_ruido_original = calcular_energia_banda(
        sinal_original,
        fs,
        f_ruido_min,
        f_ruido_max
    )

    # ------------------------------------------------------------
    # Cálculo das métricas
    # ------------------------------------------------------------

    metricas = {}

    print("\nQuestão 4.5 - Análise comparativa com os métodos anteriores")
    print("Faixa útil considerada:", faixa_util, "Hz")
    print("Faixa de ruído considerada:", faixa_ruido, "Hz")
    print("Trecho ruidoso considerado:", trecho_ruido, "s")

    for nome, sinal in sinais.items():
        trecho = sinal[idx_inicio_ruido:idx_fim_ruido]

        rms_total = calcular_rms(sinal)
        rms_trecho_ruido = calcular_rms(trecho)

        energia_util = calcular_energia_banda(
            sinal,
            fs,
            f_util_min,
            f_util_max
        )

        energia_ruido = calcular_energia_banda(
            sinal,
            fs,
            f_ruido_min,
            f_ruido_max
        )

        variacao_util_db = razao_db(energia_util, energia_util_original)
        atenuacao_ruido_db = razao_db(energia_ruido, energia_ruido_original)

        metricas[nome] = {
            "rms_total": rms_total,
            "rms_trecho_ruido": rms_trecho_ruido,
            "energia_util": energia_util,
            "energia_ruido": energia_ruido,
            "variacao_util_db": variacao_util_db,
            "atenuacao_ruido_db": atenuacao_ruido_db,
        }

        print(f"\nMétodo: {nome}")
        print("RMS total =", rms_total)
        print("RMS no trecho ruidoso =", rms_trecho_ruido)
        print("Energia na faixa útil =", energia_util)
        print("Energia na faixa de ruído =", energia_ruido)
        print("Variação da energia útil em relação ao original (dB) =", variacao_util_db)
        print("Atenuação da energia na faixa de ruído em relação ao original (dB) =", atenuacao_ruido_db)

    # ------------------------------------------------------------
    # Comparação direta entre os métodos equivalentes
    # ------------------------------------------------------------

    print("\nComparação direta entre métodos:")

    if "fft" in resultados_4:
        erro_ola_fft_vs_fft = np.max(
            np.abs(sinais["Overlap-add FFT"] - sinais["FFT"])
        )

        print("Erro máximo entre FFT direta e overlap-add FFT =", erro_ola_fft_vs_fft)
        metricas["erro_ola_fft_vs_fft"] = erro_ola_fft_vs_fft

    if "conv" in resultados_4:
        erro_ola_conv_vs_conv = np.max(
            np.abs(sinais["Overlap-add conv"] - sinais["Convolução"])
        )

        print("Erro máximo entre convolução direta e overlap-add conv =", erro_ola_conv_vs_conv)
        metricas["erro_ola_conv_vs_conv"] = erro_ola_conv_vs_conv

    if "conv" in resultados_4 and "fft" in resultados_4:
        erro_ola_conv_vs_ola_fft = np.max(
            np.abs(sinais["Overlap-add conv"] - sinais["Overlap-add FFT"])
        )

        print("Erro máximo entre overlap-add conv e overlap-add FFT =", erro_ola_conv_vs_ola_fft)
        metricas["erro_ola_conv_vs_ola_fft"] = erro_ola_conv_vs_ola_fft

    # ------------------------------------------------------------
    # Resultados da validação da Questão 4.1, caso fornecidos
    # ------------------------------------------------------------

    if validacao_4 is not None:
        print("\nValidação do overlap-add em trecho pequeno:")

        if "erro_conv_ref" in validacao_4:
            print("Erro overlap-add conv versus np.convolve =", validacao_4["erro_conv_ref"])

        if "erro_fft_ref" in validacao_4:
            print("Erro overlap-add FFT versus np.convolve =", validacao_4["erro_fft_ref"])

        if "erro_conv_fft" in validacao_4:
            print("Erro overlap-add conv versus overlap-add FFT =", validacao_4["erro_conv_fft"])

    # ------------------------------------------------------------
    # Desempenho computacional
    # ------------------------------------------------------------

    print("\nDesempenho computacional do overlap-add:")

    if "tempo_conv" in resultados_4:
        print("Tempo overlap-add por convolução =", resultados_4["tempo_conv"], "s")
        metricas["tempo_ola_conv"] = resultados_4["tempo_conv"]

    if "tempo_fft" in resultados_4:
        print("Tempo overlap-add por FFT =", resultados_4["tempo_fft"], "s")
        metricas["tempo_ola_fft"] = resultados_4["tempo_fft"]

    if "tempo_conv" in resultados_4 and "tempo_fft" in resultados_4:
        ganho = resultados_4["tempo_conv"] / resultados_4["tempo_fft"]
        print("Ganho aproximado de tempo da FFT sobre a convolução =", ganho, "vezes")
        metricas["ganho_tempo_fft"] = ganho

    # ------------------------------------------------------------
    # Análise qualitativa para o relatório
    # ------------------------------------------------------------

    print("\nAnálise qualitativa da Questão 4.5:")
    print("- O overlap-add permite processar o áudio em blocos menores, evitando a alocação de vetores muito grandes.")
    print("- A versão por FFT preserva a mesma operação de convolução linear, desde que o zero-padding seja aplicado corretamente em cada bloco.")
    print("- A saída por overlap-add FFT deve ser praticamente igual à saída por FFT direta da Questão 3.")
    print("- A versão por convolução também é equivalente, mas tende a ser mais lenta quando usa a função filtragemPorConv implementada com laços.")
    print("- A qualidade do áudio filtrado permanece praticamente a mesma, pois o filtro e a resposta ao impulso truncada são os mesmos.")
    print("- As diferenças em relação à equação de diferenças decorrem principalmente da truncagem de h[n].")
    print("- Em termos de desempenho, o overlap-add por FFT é o método mais adequado para sinais longos.")

    return metricas


def plotar_comparativo_overlap_add(metricas_4):
    """
    Plota gráficos comparativos da Questão 4.5.

    Gráficos:
        - RMS total;
        - atenuação da faixa de ruído;
        - variação da faixa útil;
        - tempo de execução, se disponível.
    """

    # Remove entradas que não são métodos, como erros e tempos
    metodos = [
        nome for nome in metricas_4.keys()
        if isinstance(metricas_4[nome], dict)
    ]

    rms_total = [metricas_4[m]["rms_total"] for m in metodos]
    atenuacao_ruido = [metricas_4[m]["atenuacao_ruido_db"] for m in metodos]
    variacao_util = [metricas_4[m]["variacao_util_db"] for m in metodos]

    # ------------------------------------------------------------
    # RMS
    # ------------------------------------------------------------

    plt.figure(num="Questão 4.5 - Comparativo RMS", figsize=(10, 4))
    plt.bar(metodos, rms_total)
    plt.title("Comparação do RMS total")
    plt.ylabel("RMS")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Atenuação na faixa de ruído
    # ------------------------------------------------------------

    plt.figure(num="Questão 4.5 - Atenuação da faixa de ruído", figsize=(10, 4))
    plt.bar(metodos, atenuacao_ruido)
    plt.title("Atenuação da energia na faixa de ruído")
    plt.ylabel("Atenuação em relação ao original (dB)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Variação da faixa útil
    # ------------------------------------------------------------

    plt.figure(num="Questão 4.5 - Variação da faixa útil", figsize=(10, 4))
    plt.bar(metodos, variacao_util)
    plt.title("Variação da energia na faixa útil")
    plt.ylabel("Variação em relação ao original (dB)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Tempo de execução, se houver
    # ------------------------------------------------------------

    tempos = {}
    
    if "tempo_ola_conv" in metricas_4:
        tempos["Overlap-add conv"] = metricas_4["tempo_ola_conv"]

    if "tempo_ola_fft" in metricas_4:
        tempos["Overlap-add FFT"] = metricas_4["tempo_ola_fft"]

    if len(tempos) > 0:
        plt.figure(num="Questão 4.5 - Tempo de execução", figsize=(8, 4))
        plt.bar(list(tempos.keys()), list(tempos.values()))
        plt.title("Tempo de execução do overlap-add")
        plt.ylabel("Tempo (s)")
        plt.grid(True, axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

def plotar_validacao_overlap_add(fs, validacao_4):
    """
    Plota a validação do overlap-add em trecho pequeno,
    comparando convolução, FFT e referência por np.convolve.
    """

    x_teste = validacao_4["x_teste"]
    y_ref = validacao_4["y_ref"]
    y_ola_conv = validacao_4["y_ola_conv"]
    y_ola_fft = validacao_4["y_ola_fft"]

    tempo_x = np.arange(len(x_teste)) / fs
    tempo_y = np.arange(len(y_ref)) / fs

    plt.figure(num="Questão 4.1 - Validação overlap-add", figsize=(12, 6))

    plt.plot(tempo_x, x_teste, label="Trecho original", linewidth=0.5)
    plt.plot(tempo_y, y_ref, label="Referência np.convolve", linewidth=0.8)
    plt.plot(tempo_y, y_ola_conv, "--", label="Overlap-add por convolução", linewidth=0.8)
    plt.plot(tempo_y, y_ola_fft, ":", label="Overlap-add por FFT", linewidth=0.8)

    plt.title("Validação do overlap-add em trecho pequeno")
    plt.xlabel("t (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()