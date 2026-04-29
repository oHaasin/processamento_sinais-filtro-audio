# Processamento e Restauração de Sinal de Áudio

Este projeto implementa técnicas de Processamento Digital de Sinais (DSP) em Python para analisar e restaurar um arquivo de áudio (audio_corrompido.wav) degradado por um ruído aleatório de espectro de faixa larga, com duração aproximada entre 16 e 26 segundos.

## Funcionalidades Implementadas

### 1. Análise Inicial
* Carregamento dos dados brutos do sinal.
* Extração e apresentação das características básicas do arquivo de áudio.

### 2. Abordagens de Filtragem
O algoritmo explora e compara três métodos matemáticos distintos para a aplicação do filtro digital:
* Equação de Diferenças: Simulação do comportamento de um conversor Analógico-Digital (A/D) no domínio do tempo.
* Convolução Direta: Filtragem no domínio do tempo utilizando a resposta ao impulso do sistema.
* Transformada de Fourier: Filtragem no domínio da frequência utilizando a propriedade da multiplicação.

### 3. Otimização de Desempenho (Overlap-Add)
A convolução direta convencional apresenta ineficiência computacional quando há uma grande disparidade de tamanho entre os blocos (o sinal de áudio é muito maior que a resposta ao impulso truncada). Para solucionar isso, o sistema foi aprimorado com:
* Implementação do algoritmo de convolução particionada Overlap-Add.
* Filtragem eficiente dividindo o sinal em blocos de tamanho Nx = Nh.

### 4. Resultados
* Avaliação comparativa do sinal antes e após a filtragem, tanto no domínio do tempo quanto no domínio da frequência.
* Integração para execução automática do áudio restaurado diretamente no sistema de som do computador.

---

## Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python instalado em sua máquina. O projeto utiliza as seguintes bibliotecas externas:
* numpy
* scipy
* matplotlib

Para instalá-las, abra o terminal (Prompt de Comando) e execute:
pip install numpy scipy matplotlib

### Passo a Passo da Execução
1. Clone este repositório para a sua máquina local:
   git clone [link_repositorio]

2. Navegue até a pasta do projeto:
   cd [pasta_repositorio]

3. Execute o script principal:
   python main.py

4. Uma interface gráfica será aberta. Selecione os arquivos solicitados na seguinte ordem:
   * O arquivo de áudio: audio_corrompido.wav
   * O arquivo do numerador do filtro: coefs_num.mat
   * O arquivo do denominador do filtro: coefs_den.mat

Os gráficos de análise serão gerados automaticamente e o áudio limpo (audio_limpo.wav) será salvo na mesma pasta do projeto.
