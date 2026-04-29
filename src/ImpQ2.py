import numpy as np
from scipy.io import loadmat


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

        # Parte não-recursiva: b[0]x[n] + b[1]x[n-1] + ...
        y_n = np.dot(self.b, self.x_hist)

        # Parte recursiva: -a[1]y[n-1] -a[2]y[n-2] - ...
        y_n -= np.dot(self.a[1:], self.y_hist)

        # Atualiza o histórico de saídas
        self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y_n

        return y_n
