import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import lfilter

from ImpQ2_2 import truncar_resposta_impulso, filtragemPorConv


def carregar_coeficiente_mat(nome_arquivo):
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


# ============================================================
# 2. Calcular resposta ao impulso com 1000 amostras
# ============================================================

N_h_total = 1000

impulso = np.zeros(N_h_total)
impulso[0] = 1.0

h = lfilter(num, den, impulso)


# ============================================================
# 3. Truncar resposta ao impulso
# ============================================================

h_trunc, Nh = truncar_resposta_impulso(h, percentual=0.01)

print("\nResposta ao impulso:")
print("Número de amostras original =", len(h))
print("Número de amostras truncado Nh =", Nh)
print("Pico de |h[n]| =", np.max(np.abs(h)))
print("Limiar de truncagem =", 0.01 * np.max(np.abs(h)))


# ============================================================
# 4. Plotar h[n] original e h_trunc[n]
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(h, label="h[n] original")
plt.plot(np.arange(Nh), h_trunc, "--", label="h_trunc[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Resposta ao impulso original e truncada")
plt.grid(True)
plt.legend()


# ============================================================
# 5. Teste com sinal aleatório pequeno
# ============================================================

np.random.seed(0)

Nx = 200
x_teste = np.random.randn(Nx)

y_conv_circular = filtragemPorConv(x_teste, h_trunc)

# Referência usando convolução pronta do NumPy
y_referencia = np.convolve(x_teste, h_trunc)

erro_max = np.max(np.abs(y_conv_circular - y_referencia))

print("\nTeste da convolução:")
print("Nx =", Nx)
print("Nh =", Nh)
print("Ny esperado =", Nx + Nh - 1)
print("Ny obtido =", len(y_conv_circular))
print("Erro máximo em relação ao np.convolve =", erro_max)


# ============================================================
# 6. Plotar comparação
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(y_conv_circular, label="Minha convolução circular")
plt.plot(y_referencia, "--", label="np.convolve")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Filtragem por convolução: implementação própria vs np.convolve")
plt.grid(True)
plt.legend()


# ============================================================
# 7. Resultado final
# ============================================================

if np.allclose(y_conv_circular, y_referencia, atol=1e-8, rtol=1e-6):
    print("\nResultado: a implementação por convolução está funcionando corretamente.")
else:
    print("\nResultado: há diferença relevante na implementação por convolução.")

plt.show()