import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import lfilter

from ImpQ2 import FiltroEqDif


def carregar_coeficiente_mat(nome_arquivo):
    """
    Carrega automaticamente o vetor de coeficientes de um arquivo .mat.
    Ignora as chaves internas do Matlab, como __header__, __version__ etc.
    """
    dados = loadmat(nome_arquivo)

    for chave, valor in dados.items():
        if not chave.startswith("__"):
            return np.asarray(valor, dtype=np.float64).flatten()

    raise ValueError(f"Nenhum coeficiente encontrado em {nome_arquivo}")


# ============================================================
# 1. Carregar coeficientes
# ============================================================

num = carregar_coeficiente_mat("coefs_num.mat")
den = carregar_coeficiente_mat("coefs_den.mat")

print("Coeficientes carregados:")
print("len(num) =", len(num))
print("len(den) =", len(den))
print("num =", num)
print("den =", den)


# ============================================================
# 2. Teste com impulso unitário
# ============================================================

N = 1000

x_impulso = np.zeros(N)
x_impulso[0] = 1.0

filtro = FiltroEqDif(num, den)

y_manual = np.zeros(N)

for n in range(N):
    y_manual[n] = filtro.filtrar_amostra(x_impulso[n])

y_scipy = lfilter(num, den, x_impulso)

erro_max = np.max(np.abs(y_manual - y_scipy))

print("\nTeste com impulso unitário:")
print("Erro máximo =", erro_max)

plt.figure(figsize=(10, 4))
plt.plot(y_manual, label="Minha implementação")
plt.plot(y_scipy, "--", label="scipy.signal.lfilter")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.title("Resposta ao impulso: implementação própria vs scipy")
plt.grid(True)
plt.legend()


# ============================================================
# 3. Teste com sinal aleatório
# ============================================================

np.random.seed(0)

x_teste = np.random.randn(5000)

filtro.resetar()

y_manual = np.zeros_like(x_teste)

for n in range(len(x_teste)):
    y_manual[n] = filtro.filtrar_amostra(x_teste[n])

y_scipy = lfilter(num, den, x_teste)

erro_max = np.max(np.abs(y_manual - y_scipy))

print("\nTeste com sinal aleatório:")
print("Erro máximo =", erro_max)

plt.figure(figsize=(10, 4))
plt.plot(y_manual[:300], label="Minha implementação")
plt.plot(y_scipy[:300], "--", label="scipy.signal.lfilter")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Sinal aleatório: implementação própria vs scipy")
plt.grid(True)
plt.legend()
plt.show()


# ============================================================
# 4. Critério de aprovação
# ============================================================

if np.allclose(y_manual, y_scipy, atol=1e-8, rtol=1e-6):
    print("\nResultado: a implementação está funcionando corretamente.")
else:
    print("\nResultado: há diferença relevante entre a implementação e o scipy.")