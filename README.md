<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/OpenCV-Webcam-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
<img src="https://img.shields.io/badge/Optuna-HPO-2C3E50?style=for-the-badge"/>
<img src="https://img.shields.io/badge/YOLOv11n--face-Detecção-111F68?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Status-Fases%201%20e%202%20Concluídas-brightgreen?style=for-the-badge"/>

<br/><br/>

# 👁️ Visão Computacional — Pessoas e Expressões Faciais

### Classificação de pessoas, detecção multi-face e reconhecimento de expressões faciais em imagens e vídeo

**Universidade Federal do Maranhão (UFMA)** · Grupo 05 · 2026

</div>

---

## 👥 Equipe

| Nome | GitHub |
|------|--------|
| Caio Bandeira | — |
| Davih Asaph | — |
| José Marques | — |
| Maurício Miranda | — |

---

## 🗺️ Roadmap do Projeto

```
✅ Fase 1 — Reconhecimento de pessoas numa imagem            [CONCLUÍDA]
           Classificação binária com Transfer Learning
           (AlexNet · ResNet18 · EfficientNet-B0)

✅ Fase 2 — Detecção facial e reconhecimento de expressões   [CONCLUÍDA]
           CNN inspirada na AnyNet + otimização com Optuna
           Detecção de múltiplos rostos com YOLOv11n-face
           Aplicação em tempo real com webcam
```

> **Atualização do enunciado:** na versão mais recente da atividade, a detecção dos rostos e o reconhecimento das expressões foram reunidos na **Fase 2**. Por isso, a antiga Fase 3 não aparece separadamente nesta README.

---

# 📦 Fase 1 — Classificação Binária com Transfer Learning

> **Objetivo:** Dado uma imagem, determinar se ela contém uma pessoa ou não.

## 🗂️ Dataset — Pascal VOC 2012

O **Pascal VOC 2012** é um benchmark clássico de visão computacional com 20 classes de objetos. Para adaptar ao problema binário, as anotações XML foram processadas para gerar dois grupos:

| Classe | Imagens | Proporção |
|--------|---------|-----------|
| **Com Pessoa** | 9.583 | 56,0% |
| **Sem Pessoa** | 7.542 | 44,0% |
| **Total** | **17.125** | 100% |

O dataset apresenta um leve desbalanceamento favorável à classe "pessoa", típico de datasets de cenas do mundo real. O uso de validação cruzada estratificada garante que essa proporção se mantenha em todos os folds.

---

## 🧠 Arquiteturas Utilizadas

### AlexNet (2012)
> Krizhevsky et al. — a CNN que popularizou o deep learning em visão computacional

A AlexNet usa convoluções empilhadas com ReLU, max-pooling e dropout no classificador. Sua estrutura sequencial de 5 blocos convolucionais a torna um ponto de partida clássico para transfer learning.

```
features[0–4]  : conv1 (11×11, stride 4) + ReLU + MaxPool → conv2 (5×5) + ReLU + MaxPool
features[5–9]  : conv3 (3×3) + ReLU → conv4 (3×3) + ReLU
features[10–12]: conv5 (3×3) + ReLU + MaxPool  ← alvo do Grad-CAM
avgpool → Dropout → FC(4096) → Dropout → FC(4096) → FC(2)  [cabeça binária]

Parâmetros totais: 57.012.034
```

---

### ResNet18 (2015)
> He et al. — residual learning para redes mais profundas sem degradação

A ResNet18 usa **skip connections** que permitem ao gradiente fluir mais facilmente pelas camadas, mitigando o problema de degradação de redes profundas. É a variante mais leve da família ResNet.

```
conv1 (7×7, stride 2) + BN + MaxPool
layer1: 2× BasicBlock (64 filtros, conv 3×3)
layer2: 2× BasicBlock (128 filtros, conv 3×3, stride 2)
layer3: 2× BasicBlock (256 filtros, conv 3×3, stride 2)
layer4: 2× BasicBlock (512 filtros, conv 3×3, stride 2)  ← alvo do Grad-CAM
AdaptiveAvgPool → FC(2)

Parâmetros totais: 11.177.538
```

---

### EfficientNet-B0 (2019)
> Tan & Le — compound scaling de largura, profundidade e resolução

A EfficientNet usa **MBConv blocks** (convoluções depthwise separáveis com squeeze-and-excitation) e um princípio de escalonamento composto. É significativamente mais eficiente que ResNets e AlexNet para acurácia equivalente ou superior.

```
stem: conv 3×3 (32 filtros, stride 2)
MBConv blocks (1→6×): 7 estágios com expansão crescente
                       squeeze-and-excitation em cada bloco
conv head + BN + SiLU
AdaptiveAvgPool → Dropout → FC(2)

Parâmetros totais: 4.010.110  (8× menos que ResNet18, 14× menos que AlexNet)
```

---

## 🔧 Regimes de Transfer Learning

Três abordagens foram comparadas para cada arquitetura:

| Regime | O que treina | Parâmetros treináveis |
|--------|-------------|----------------------|
| **Frozen** | Apenas a cabeça classificadora | ~0,0–0,1% |
| **Shallow FT** | Últimas camadas + classificador | 75–97% |
| **Full FT** | Rede inteira (com CosineAnnealing) | 100% |

### Estratégia de Fine-tuning

**Frozen** — o backbone ImageNet é completamente congelado. As features genéricas (bordas, texturas, gradientes) extraídas do ImageNet são usadas diretamente. Rápido de treinar, mas limitado na adaptação ao domínio.

**Shallow Fine-tuning** — descongela apenas as camadas finais do backbone (ex: `layer4` na ResNet, `blocks[-2:]` na EfficientNet). As camadas iniciais mantêm features genéricas; as finais se adaptam ao domínio de pessoas.

**Full Fine-tuning** — toda a rede treina com taxa de aprendizado baixa e scheduler cosine annealing. Maior capacidade de adaptação, mas requer regularização cuidadosa para evitar overfitting.

---

## ⚙️ Configuração Experimental

```python
# Protocolo de validação
N_FOLDS    = 5       # StratifiedKFold — mantém proporção de classes
VAL_SPLIT  = 0.10    # 10% dos dados de treino do fold → validação interna
NUM_EPOCHS = 10      # épocas por fold por experimento

# Cada fold:
# ├─ Treino:       ~12.330 imagens
# ├─ Validação:    ~1.370 imagens  (checkpoint pelo melhor F1)
# └─ Teste:        ~3.425 imagens  (avaliação final)

# Pré-processamento
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]   # ImageNet mean
STD  = [0.229, 0.224, 0.225]   # ImageNet std

# Data augmentation (treino apenas)
RandomHorizontalFlip()
ColorJitter(brightness=0.2, contrast=0.2)

# Otimizador: Adam  |  Loss: CrossEntropyLoss
# Scheduler: CosineAnnealingLR (Full FT)
# Device: GPU Tesla T4 (Kaggle)
```

---

## 📊 Resultados — Tabela Completa por Fold

### AlexNet

| Regime | Fold | Acurácia | Precisão (P) | Recall (P) | F1 (P) |
|--------|------|----------|-------------|------------|--------|
| Frozen | 1 | 82,69% | 83,07% | 86,75% | 84,87% |
| Frozen | 2 | 82,66% | 80,55% | 90,98% | 85,45% |
| Frozen | 3 | 84,61% | 84,54% | 88,73% | 86,59% |
| Frozen | 4 | 83,88% | 84,90% | 86,59% | 85,74% |
| Frozen | 5 | 83,97% | 82,97% | 89,77% | 86,24% |
| **Frozen Média** | — | **83,56%** | 83,21% | 88,56% | **85,78%** |
| Shallow FT | 1 | 87,39% | 89,64% | 87,58% | 88,60% |
| Shallow FT | 2 | 88,79% | 90,88% | 88,89% | 89,87% |
| Shallow FT | 3 | 88,26% | 91,83% | 86,75% | 89,22% |
| Shallow FT | 4 | 87,74% | 89,12% | 88,94% | 89,03% |
| Shallow FT | 5 | 88,12% | 90,07% | 88,52% | 89,29% |
| **Shallow FT Média** | — | **88,06%** | 90,31% | 88,14% | **89,20%** |
| Full FT | 1 | 87,27% | 89,83% | 87,12% | 88,45% |
| Full FT | 2 | 88,64% | 90,47% | 89,10% | 89,78% |
| Full FT | 3 | 88,20% | 92,57% | 85,81% | 89,06% |
| Full FT | 4 | 88,55% | 91,87% | 87,27% | 89,51% |
| Full FT | 5 | 88,38% | 91,48% | 87,37% | 89,38% |
| **Full FT Média** | — | **88,21%** | 91,24% | 87,33% | **89,24%** |

### ResNet18

| Regime | Fold | Acurácia | Precisão (P) | Recall (P) | F1 (P) |
|--------|------|----------|-------------|------------|--------|
| Frozen | 1 | 88,61% | 95,04% | 84,04% | 89,20% |
| Frozen | 2 | 90,01% | 95,03% | 86,70% | 90,67% |
| Frozen | 3 | 89,20% | 94,63% | 85,55% | 89,86% |
| Frozen | 4 | 90,42% | 94,86% | 87,63% | 91,10% |
| Frozen | 5 | 89,05% | 93,31% | 86,64% | 89,85% |
| **Frozen Média** | — | **89,46%** | 94,57% | 86,11% | **90,14%** |
| Shallow FT | 1 | 91,80% | 95,49% | 89,57% | 92,44% |
| Shallow FT | 2 | 92,44% | 96,63% | 89,62% | 92,99% |
| Shallow FT | 3 | 91,77% | 95,44% | 89,57% | 92,41% |
| Shallow FT | 4 | 92,32% | 95,69% | 90,34% | 92,94% |
| Shallow FT | 5 | 91,62% | 94,88% | 89,87% | 92,31% |
| **Shallow FT Média** | — | **91,99%** | 95,63% | 89,79% | **92,62%** |
| Full FT | 1 | 92,12% | 97,08% | 88,58% | 92,64% |
| Full FT | 2 | 93,31% | 97,20% | 90,66% | 93,82% |
| Full FT | 3 | 92,41% | 96,62% | 89,57% | 92,96% |
| Full FT | 4 | 92,82% | 96,08% | 90,87% | 93,40% |
| Full FT | 5 | 92,23% | 95,78% | 90,08% | 92,85% |
| **Full FT Média** | — | **92,58%** | 96,55% | 89,95% | **93,13%** |

### EfficientNet-B0

| Regime | Fold | Acurácia | Precisão (P) | Recall (P) | F1 (P) |
|--------|------|----------|-------------|------------|--------|
| Frozen | 1 | 87,85% | 92,86% | 84,82% | 88,66% |
| Frozen | 2 | 89,02% | 92,97% | 86,96% | 89,87% |
| Frozen | 3 | 87,80% | 90,32% | 87,58% | 88,93% |
| Frozen | 4 | 89,69% | 93,30% | 87,89% | 90,51% |
| Frozen | 5 | 87,80% | 88,77% | 89,51% | 89,14% |
| **Frozen Média** | — | **88,43%** | 91,64% | 87,35% | **89,42%** |
| Shallow FT | 1 | 92,73% | 95,72% | 91,08% | 93,34% |
| Shallow FT | 2 | 93,78% | 96,86% | 91,86% | 94,30% |
| Shallow FT | 3 | 92,96% | 95,99% | 91,24% | 93,55% |
| Shallow FT | 4 | 93,90% | 96,41% | 92,54% | 94,43% |
| Shallow FT | 5 | 92,79% | 96,70% | 90,19% | 93,33% |
| **Shallow FT Média** | — | **93,23%** | 96,34% | 91,38% | **93,79%** |
| Full FT | 1 | 93,11% | 95,21% | 92,33% | 93,75% |
| Full FT | 2 | 94,45% | 97,89% | 92,07% | 94,89% |
| Full FT | 3 | 93,93% | 96,72% | 92,28% | 94,45% |
| Full FT | 4 | 94,54% | 97,37% | 92,75% | 95,00% |
| Full FT | 5 | 93,49% | 95,15% | 93,11% | 94,12% |
| **Full FT Média** | — | **93,90%** | 96,47% | 92,51% | **94,44%** |

---

## 🏆 Resumo Comparativo — Melhor por Modelo

| Modelo | Regime | Acurácia | F1 (Pessoa) | Especificidade |
|--------|--------|----------|-------------|---------------|
| AlexNet | Full FT | 88,21% ± 0,55% | 89,24% ± 0,51% | 89,33% ± 1,55% |
| AlexNet | Shallow FT | 88,06% ± 0,53% | 89,20% ± — | 87,96% ± 1,52% |
| ResNet18 | Full FT | 92,58% ± 0,49% | 93,13% ± 0,47% | 95,91% ± 0,77% |
| ResNet18 | Shallow FT | 91,99% ± 0,36% | 92,62% ± — | 94,78% ± 0,79% |
| **EfficientNet-B0** | **Full FT** | **93,90% ± 0,61%** | **94,44% ± 0,52%** | **95,68% ± 1,59%** |
| EfficientNet-B0 | Shallow FT | 93,23% ± 0,56% | 93,79% ± — | 95,59% ± 0,59% |

> 🥇 **EfficientNet-B0 com Full Fine-Tuning** obteve o melhor desempenho geral: **93,90% de acurácia** e **94,44% de F1** na classe pessoa — com apenas 4 milhões de parâmetros contra 57 milhões da AlexNet.

---

## 🔍 Grad-CAM — Interpretabilidade Visual

O **Grad-CAM** (Gradient-weighted Class Activation Mapping) gera mapas de calor que destacam quais regiões da imagem o modelo "olhou" para tomar sua decisão.

### Como funciona

```
1. Forward pass na imagem → activations da última camada conv
2. Backward pass para a classe predita → gradients
3. Global Average Pooling dos gradients → pesos de importância por canal
4. Soma ponderada dos activation maps → heatmap bruto
5. ReLU + upscale + sobreposição na imagem original
```

### Camada alvo por arquitetura

| Modelo | Camada alvo |
|--------|-------------|
| AlexNet | `features[10]` — última convolução do bloco 5 |
| ResNet18 | `layer4[-1].conv2` — última convolução do último bloco residual |
| EfficientNet-B0 | Último bloco MBConv (`blocks[-1]`) |

### Estratégia de seleção

Para o Grad-CAM, o fold com **maior F1 no teste** de cada experimento foi selecionado automaticamente — garantindo que os mapas de calor reflitam o modelo no seu melhor estado generalizado. As mesmas imagens são usadas para todos os modelos, permitindo comparação justa.

> Os mapas de calor mostram que as redes com fine-tuning (Shallow e Full) focam consistentemente na **silhueta e postura humana**, enquanto a Frozen tende a ativar regiões mais dispersas, evidenciando a limitação de features ImageNet não adaptadas.

---

# 🎭 Fase 2 — Detecção Facial e Reconhecimento de Expressões

> **Objetivo:** detectar todos os rostos presentes em uma imagem ou quadro de vídeo e classificar a expressão facial de cada pessoa individualmente.

A segunda fase integra duas tarefas de visão computacional em um único fluxo:

1. **Detecção facial**, responsável por localizar cada rosto e gerar sua *bounding box*;
2. **Classificação de expressão**, responsável por analisar o recorte de cada rosto e atribuir uma emoção.

O detector permanece fixo como **YOLOv11n-face**, enquanto o modelo usado para inferência de emoções pode ser alterado pela própria interface da aplicação.

---

## 🗂️ Dataset — FERPlus

O treinamento do classificador de emoções foi realizado com o **FERPlus**, dataset de expressões faciais recomendado para a atividade. As imagens passam pelas rotinas de pré-processamento e *data augmentation* definidas no notebook antes de serem enviadas à CNN.

### Classes de expressão

| Classe | Exibição na aplicação | Emoji |
|--------|-----------------------|-------|
| Neutral | Neutro | 😐 |
| Happiness | Feliz | 😀 |
| Surprise | Surpreso | 😮 |
| Sadness | Triste | 😢 |
| Anger | Raiva | 😠 |
| Disgust | Nojo | 🤢 |
| Fear | Medo | 😨 |
| Contempt | Desprezo | 😒 |

> A ordem interna das classes deve permanecer igual à utilizada durante o treinamento e salva no checkpoint do classificador.

---

## 🧠 CNN Inspirada na AnyNet

Em vez de utilizar somente uma arquitetura clássica pronta, foi construída uma CNN seguindo os conceitos de **estimação e organização de arquiteturas da AnyNet**, apresentados no capítulo de projeto de CNNs modernas do *Dive into Deep Learning*.

A rede é organizada em componentes configuráveis:

```text
Imagem facial
    ↓
Stem convolucional
    ↓
Estágio 1 — blocos convolucionais
    ↓
Estágio 2 — aumento de canais e redução espacial
    ↓
Estágio 3 — extração de características de maior nível
    ↓
Global Average Pooling
    ↓
Dropout + camada totalmente conectada
    ↓
Probabilidades das expressões faciais
```

Entre os elementos avaliados durante os experimentos estão:

- quantidade de estágios e blocos;
- número de canais por estágio;
- regularização por dropout e weight decay;
- taxa de aprendizado e otimizador;
- tamanho de lote e estratégia de treinamento;
- desempenho médio em validação cruzada.

---

## 🔎 Otimização com Optuna

O **Optuna** foi utilizado para automatizar a busca de configurações da arquitetura e do treinamento. Cada *trial* representa uma configuração completa, avaliada com **validação cruzada estratificada de 5 folds**.

```text
Trial do Optuna
    ├── Fold 1: treino → validação → métricas
    ├── Fold 2: treino → validação → métricas
    ├── Fold 3: treino → validação → métricas
    ├── Fold 4: treino → validação → métricas
    └── Fold 5: treino → validação → métricas
             ↓
      F1-score médio do trial
             ↓
   Seleção da melhor configuração
```

Durante o treinamento são registrados, por época e por fold:

- loss de treino e validação;
- acurácia;
- precisão, recall e F1-score;
- melhor checkpoint;
- estado do estudo do Optuna;
- informações necessárias para retomar uma execução interrompida.

<!-- Adicione aqui o link do projeto no Weights & Biases caso ele faça parte da versão entregue. -->

---

## 🎯 Detecção Facial com YOLOv11n-face

A aplicação utiliza exclusivamente o **YOLOv11n-face** para detectar os rostos. O detector é executado uma vez em cada quadro da webcam e retorna todas as faces encontradas, em vez de analisar apenas a maior face da cena.

Para cada detecção, a aplicação:

```text
Frame da webcam
    ↓
YOLOv11n-face
    ↓
Bounding boxes de todos os rostos
    ↓
Recorte individual de cada face
    ↓
Pré-processamento do classificador
    ↓
CNN de expressão facial
    ↓
Classe + confiança + emoji
    ↓
Resultado desenhado no frame
```

O uso de um detector separado do classificador permite que cada componente seja atualizado sem alterar o restante do pipeline.

---

## 🖥️ Aplicação em Tempo Real

A aplicação final foi desenvolvida para trabalhar com webcam e realizar inferência em tempo real.

### Funcionalidades implementadas

- captura contínua dos quadros da webcam;
- detecção simultânea de múltiplos rostos;
- classificação independente da expressão de cada pessoa;
- bounding box com emoção, confiança e emoji correspondente;
- HUD com FPS, quantidade de faces e informações de execução;
- exibição do **modelo de classificação de emoções** atualmente selecionado;
- troca do modelo de inferência emocional sem alterar o YOLOv11n-face;
- carregamento dos emojis por arquivos PNG;
- controle de limiares de confiança da detecção e da classificação.

### Separação de responsabilidades

| Componente | Responsabilidade |
|------------|------------------|
| `YOLOv11n-face` | Encontrar e delimitar todos os rostos |
| Classificador de emoções | Determinar a expressão de cada recorte facial |
| OpenCV | Capturar a webcam e desenhar a interface |
| Emojis PNG | Representar visualmente a emoção prevista |
| HUD | Exibir modelo, FPS, faces e parâmetros ativos |

---

## 📊 Avaliação da Fase 2

A avaliação do classificador é realizada separadamente da aplicação em tempo real. Os notebooks armazenam resultados por fold, matrizes de confusão, curvas de treinamento e métricas agregadas.

As métricas consolidadas da configuração final devem ser obtidas diretamente dos artefatos gerados pelo treinamento. Elas não foram copiadas para esta README sem os arquivos finais de resultados, evitando registrar números incompletos ou de execuções interrompidas.

### Artefatos produzidos

- estudo do Optuna e histórico dos trials;
- checkpoints por fold e melhor checkpoint global;
- métricas por classe e médias macro;
- matrizes de confusão;
- curvas de loss e desempenho;
- aplicação de webcam com inferência multi-face;
- detector `yolov11n-face.pt` e classificadores de emoção.

---

## 🧩 Principais Desafios Técnicos

- manter a correspondência correta entre índices, nomes e emojis das emoções;
- lidar com o desbalanceamento natural entre as classes do FERPlus;
- evitar vazamento de dados durante a validação cruzada;
- preservar checkpoints para continuar treinamentos interrompidos;
- executar detecção e classificação com latência suficiente para webcam;
- recortar rostos próximos às bordas sem gerar regiões inválidas;
- suportar várias pessoas no mesmo frame;
- separar a troca do classificador de emoções do detector facial.

---

## 📁 Estrutura do Repositório

```text
visao-computacional/
│
├── fase-1/                              # Classificação binária: pessoa vs. não pessoa
│   ├── alexnetexperiment.ipynb          # AlexNet — 3 regimes × 5 folds + Grad-CAM
│   ├── resnet18.ipynb                   # ResNet18 — 3 regimes × 5 folds + Grad-CAM
│   ├── efficientnet-b0.ipynb            # EfficientNet-B0 — 3 regimes × 5 folds + Grad-CAM
│   └── minirelatorio.txt                # Mini-relatório da fase
│
├── fase-2/                              # Expressões faciais e aplicação com webcam
│   ├── notebooks/                       # Treinamento, validação cruzada e Optuna
│   ├── checkpoints/                     # Pesos dos classificadores de emoção
│   ├── resultados/                      # Métricas, matrizes e curvas
│   ├── assets/
│   │   └── emojis/                      # Emojis em PNG usados na interface
│   ├── models/
│   │   └── yolov11n-face.pt             # Detector facial fixo
│   └── face_detection_yolonas_comp.py   # Aplicação de webcam e inferência multi-face
│
└── README.md
```

> Os nomes das subpastas podem ser ajustados à organização final do repositório, mantendo a separação entre treinamento, pesos, resultados e aplicação.

---

## 🔬 Análise Crítica

### Pontos Fortes

- **Protocolo rigoroso na Fase 1:** validação cruzada estratificada com 5 folds e comparação sistemática de três arquiteturas e três regimes de transfer learning.
- **Diversidade arquitetural:** AlexNet, ResNet18 e EfficientNet-B0 representam abordagens clássica, residual e eficiente.
- **Interpretabilidade:** Grad-CAM foi utilizado para verificar as regiões relevantes para a decisão dos classificadores.
- **Arquitetura própria na Fase 2:** a CNN de emoções foi montada com conceitos de projeto da AnyNet, em vez de depender apenas de uma rede pronta.
- **Otimização automatizada:** Optuna foi integrado ao protocolo de validação cruzada para comparar configurações completas.
- **Pipeline modular:** detecção facial e classificação emocional são componentes independentes.
- **Aplicação multi-face:** todas as pessoas detectadas no ambiente são processadas individualmente.
- **Continuidade de treinamento:** checkpoints e estados de execução reduzem a perda de progresso em sessões interrompidas.

### Limitações

- classes menos frequentes do FERPlus podem apresentar maior variabilidade de desempenho;
- iluminação, oclusões, rotação da cabeça e baixa resolução podem afetar a inferência em webcam;
- a emoção facial prevista representa um padrão visual e não deve ser interpretada como diagnóstico do estado emocional real da pessoa;
- a velocidade depende da câmera, do hardware e da quantidade de rostos no frame;
- os resultados da detecção e da classificação devem ser avaliados separadamente para identificar a origem dos erros.

### Possíveis Evoluções

- adicionar explicabilidade ao classificador de emoções com Grad-CAM;
- calibrar probabilidades e limiares por classe;
- testar técnicas adicionais para desbalanceamento;
- exportar os modelos para ONNX ou TensorRT;
- adicionar processamento de vídeos gravados e imagens estáticas;
- comparar novos classificadores mantendo o YOLOv11n-face fixo;
- criar relatórios automáticos com métricas e exemplos de acertos e erros.

---

## 🛠️ Como Reproduzir

### Pré-requisitos gerais

```bash
pip install torch torchvision timm torchmetrics scikit-learn \
    matplotlib seaborn pandas numpy pillow opencv-python \
    ultralytics optuna
```

Caso a versão entregue utilize o rastreamento online previsto no enunciado:

```bash
pip install wandb
```

### Fase 1 — Pascal VOC 2012

1. Baixe o **Pascal VOC 2012** em: http://host.robots.ox.ac.uk/pascal/VOC
2. Extraia o dataset para o caminho configurado nos notebooks.
3. Execute os experimentos:

```bash
jupyter nbconvert --to notebook --execute alexnetexperiment.ipynb
jupyter nbconvert --to notebook --execute resnet18.ipynb
jupyter nbconvert --to notebook --execute efficientnet-b0.ipynb
```

### Fase 2 — FERPlus e Optuna

1. Baixe o **FERPlus** em: https://www.kaggle.com/datasets/arnabkumarroy02/ferplus
2. Ajuste no notebook os caminhos do dataset, checkpoints e resultados.
3. Execute o treinamento em ambiente com GPU.
4. Preserve o banco/estado do Optuna e os checkpoints para permitir retomada.
5. Copie o melhor classificador para a pasta de modelos da aplicação.

### Aplicação com webcam

Certifique-se de que os seguintes arquivos estão disponíveis nos caminhos configurados:

```text
models/yolov11n-face.pt
checkpoints/<modelo_de_emocoes>.pt
assets/emojis/<emocoes>.png
```

Execute:

```bash
python face_detection_yolonas_comp.py
```

> Para a aplicação local, permita o acesso à câmera e confirme que o índice configurado da webcam está correto. Para treinamento e otimização, recomenda-se utilizar o Kaggle com GPU habilitada.

---

## 📚 Referências

### Fase 1

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Everingham, M., et al. (2010). *The Pascal Visual Object Classes (VOC) Challenge.* IJCV.

### Fase 2

- Barsoum, E., Zhang, C., Ferrer, C. C., & Zhang, Z. (2016). *Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution.*
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. *Dive into Deep Learning — Designing Convolution Network Architectures.* https://d2l.ai/chapter_convolutional-modern/cnn-design.html
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD.
- Optuna. https://optuna.org/
- Weights & Biases. https://wandb.ai/site/
- FERPlus Dataset. https://www.kaggle.com/datasets/arnabkumarroy02/ferplus
- Ultralytics YOLO Documentation. https://docs.ultralytics.com/

---

<div align="center">

**Fase 1 concluída em 08/05/2026 · Fase 2 concluída em 2026**

*Projeto desenvolvido para a disciplina de Visão Computacional — 2026*

</div>
