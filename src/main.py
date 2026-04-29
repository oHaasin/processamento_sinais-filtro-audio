import carregamento_dados_1
import filtros_2
import filtragem_3
import bonus_4

# Abre a janela e pega o caminho do arquivo
caminho = carregamento_dados_1.carregar_wav()

# Plota o gráfico de tempo e recebe de volta a taxa e os dados do áudio
taxa, dados = carregamento_dados_1.plotar_sinal_tempo(caminho)

# Usa a taxa e os dados recebidos acima para gerar o espectro de frequência
carregamento_dados_1.plotar_espectro_frequencia(dados, taxa)

