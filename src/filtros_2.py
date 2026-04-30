import numpy as np

#! ========================================
#! 2.1 Filtragem pela equação de diferenças
#! ========================================
class FiltroEqDif:
    """
    Filtro IIR implementado pela equação de diferenças:

    y[n] = sum(b[k] x[n-k]) - sum(a[k] y[n-k]), para k >= 1

    A função filtrar_amostra(x_n) simula o comportamento pedido:
    recebe uma amostra por vez e retorna uma amostra de saída.
    """

    def __init__(self, b, a):
        self.b = np.asarray(b, dtype=np.float64).flatten()
        self.a = np.asarray(a, dtype=np.float64).flatten()

        if self.a[0] == 0:
            raise ValueError("O coeficiente a[0] não pode ser zero.")

        # Normaliza os coeficientes caso a[0] não seja 1
        self.b = self.b / self.a[0]
        self.a = self.a / self.a[0]

        # Históricos iniciais zerados
        self.x_hist = np.zeros(len(self.b), dtype=np.float64)
        self.y_hist = np.zeros(len(self.a) - 1, dtype=np.float64)

    def resetar(self):
        """
        Reinicia os históricos do filtro.
        Use antes de filtrar um novo sinal.
        """
        self.x_hist[:] = 0.0
        self.y_hist[:] = 0.0

    def filtrar_amostra(self, x_n):
        """
        Recebe uma amostra x[n] e retorna a saída correspondente y[n].
        """

        # Desloca o histórico de entradas:
        # x[n-1] vai para x[n-2], x[n-2] vai para x[n-3], etc.
        self.x_hist[1:] = self.x_hist[:-1]

        # Insere a amostra atual
        self.x_hist[0] = x_n

        # Parte não-recursiva:
        # b[0]x[n] + b[1]x[n-1] + ...
        y_n = np.dot(self.b, self.x_hist)

        # Parte recursiva:
        # -a[1]y[n-1] - a[2]y[n-2] - ...
        y_n -= np.dot(self.a[1:], self.y_hist)

        # Atualiza o histórico de saídas
        self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y_n

        return y_n

#! ================================================
#! 2.2 Cálculo e truncamento da resposta ao impulso
#! ================================================

def truncar_resposta_impulso(h, percentual=0.01):
    """
    Trunca a resposta ao impulso eliminando amostras da cauda
    menores que percentual do valor de pico.

    As amostras iniciais são mantidas, mesmo que sejam menores
    que 1% do pico. Por isso, o corte é feito apenas na cauda.

    Parâmetros:
        h: resposta ao impulso completa
        percentual: percentual do pico usado como limiar

    Retorna:
        h_trunc: resposta ao impulso truncada
        Nh: número de amostras de h_trunc
    """

    h = np.asarray(h, dtype=np.float64).flatten()

    pico = np.max(np.abs(h))
    limiar = percentual * pico

    indices_significativos = np.where(np.abs(h) >= limiar)[0]

    if len(indices_significativos) == 0:
        h_trunc = h[:1]
    else:
        ultimo_indice = indices_significativos[-1]
        h_trunc = h[:ultimo_indice + 1]

    Nh = len(h_trunc)

    return h_trunc, Nh

#! =====================================
#! 2.3 Filtragem por convolução circular
#! =====================================

def filtragemPorConv(x, h_trunc):
    """
    Filtragem por convolução circular com zero-padding.

    A função recebe o bloco completo x[n] e retorna y[n]
    de tamanho Ny = Nx + Nh - 1.

    Parâmetros:
        x: sinal de entrada
        h_trunc: resposta ao impulso truncada

    Retorna:
        y: sinal filtrado
    """

    x = np.asarray(x, dtype=np.float64).flatten()
    h_trunc = np.asarray(h_trunc, dtype=np.float64).flatten()

    Nx = len(x)
    Nh = len(h_trunc)
    Ny = Nx + Nh - 1

    # Zero-padding para transformar a convolução circular
    # em convolução linear equivalente
    x_pad = np.zeros(Ny, dtype=np.float64)
    h_pad = np.zeros(Ny, dtype=np.float64)

    x_pad[:Nx] = x
    h_pad[:Nh] = h_trunc

    y = np.zeros(Ny, dtype=np.float64)

    # Convolução circular de tamanho Ny
    for n in range(Ny):
        soma = 0.0

        for k in range(Ny):
            soma += h_pad[k] * x_pad[(n - k) % Ny]

        y[n] = soma

    return y

#! =====================
#! 2.4 Filtragem por FFT
#! =====================

def filtragemPorFFT(x, h_trunc):
    """
    Filtragem usando a propriedade da multiplicação da FFT.

    y[n] = iFFT(FFT(x[n]) * FFT(h_trunc[n]))

    A função recebe um bloco completo x[n] e retorna y[n]
    de tamanho Ny = Nx + Nh - 1.

    Parâmetros:
        x: sinal de entrada
        h_trunc: resposta ao impulso truncada

    Retorna:
        y: sinal filtrado
    """

    x = np.asarray(x, dtype=np.float64).flatten()
    h_trunc = np.asarray(h_trunc, dtype=np.float64).flatten()

    Nx = len(x)
    Nh = len(h_trunc)
    Ny = Nx + Nh - 1

    # Zero-padding até Ny para evitar aliasing circular
    x_pad = np.zeros(Ny, dtype=np.float64)
    h_pad = np.zeros(Ny, dtype=np.float64)

    x_pad[:Nx] = x
    h_pad[:Nh] = h_trunc

    # Transformadas de Fourier
    X = np.fft.fft(x_pad)
    H = np.fft.fft(h_pad)

    # Multiplicação no domínio da frequência
    Y = X * H

    # Retorno ao domínio do tempo
    y = np.fft.ifft(Y)

    # Remove resíduos imaginários numéricos
    y = np.real(y)

    return y