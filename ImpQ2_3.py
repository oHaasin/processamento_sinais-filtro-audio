import numpy as np


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