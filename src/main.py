"""
# Autores: Oliver Haas, Gabriel Paiva
# Data de última modificação: 29/04/2026
#? Objetivo: Juntar as partes do trabalho
#todo: Terminar primeira parte, adicionar demais
"""

import carregamento_dados_1
import filtros_2
import filtragem_3
import bonus_4

#* Execução da primeira parte

# Abre a janela e pega o caminho do arquivo
caminho = carregamento_dados_1.carregar_wav()

# Plota o gráfico de tempo x(t) e recebe de volta a taxa e os dados do áudio
fs, dados = carregamento_dados_1.plotar_sinal_tempo(caminho)

# Usa a taxa e os dados recebidos acima para plotar X(e^jw)
carregamento_dados_1.plotar_espectro_frequencia(fs, dados)

# Abre a janela para carregar os coeficientes
numerador, denominador = carregamento_dados_1.carrega_mat()

# Passa os parâmetros e gera o gráfico de H(e^jw)
carregamento_dados_1.plotar_resposta_frequencia(numerador, denominador, fs)

# Gera a resposta ao impulso h[n] com 1000 amostras
h = carregamento_dados_1.plotar_resposta_impulso(numerador, denominador, n_amostras=1000)

#* Execução da segunda parte

#todo

#* Execução da terceira parte

#todo

#* Execução da quarta parte
