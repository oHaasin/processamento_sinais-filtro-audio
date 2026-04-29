import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import lfilter

from ImpQ2_2 import truncar_resposta_impulso
from ImpQ2_3 import filtragemPorFFT


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

print("\nResposta ao impulso truncada:")
print("Nh =", Nh)
print("Pico de |h[n]| =", np.max(np.abs(h)))
print("Limiar de truncagem =", 0.01 * np.max(np.abs(h)))


# ============================================================
# 4. Teste com sinal aleatório
# ============================================================

np.random.seed(0)

Nx = 500
x_teste = np.random.randn(Nx)

y_fft = filtragemPorFFT(x_teste, h_trunc)

# Referência usando convolução direta pronta
y_referencia = np.convolve(x_teste, h_trunc)

erro_max = np.max(np.abs(y_fft - y_referencia))

print("\nTeste da filtragem por FFT:")
print("Nx =", Nx)
print("Nh =", Nh)
print("Ny esperado =", Nx + Nh - 1)
print("Ny obtido =", len(y_fft))
print("Erro máximo em relação ao np.convolve =", erro_max)


# ============================================================
# 5. Plotar comparação
# ============================================================

plt.figure(figsize=(10, 4))
plt.plot(y_fft, label="Minha filtragem por FFT")
plt.plot(y_referencia, "--", label="np.convolve")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.title("Filtragem por FFT: implementação própria vs np.convolve")
plt.grid(True)
plt.legend()


# ============================================================
# 6. Resultado final
# ============================================================

if np.allclose(y_fft, y_referencia, atol=1e-8, rtol=1e-6):
    print("\nResultado: a implementação por FFT está funcionando corretamente.")
else:
    print("\nResultado: há diferença relevante na implementação por FFT.")

plt.show()