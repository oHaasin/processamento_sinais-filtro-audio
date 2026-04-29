import numpy as np


def truncar_resposta_impulso(h, percentual=0.01):
    """
    Trunca a resposta ao impulso eliminando amostras da cauda
    menores que percentual do valor de pico.

    Importante:
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