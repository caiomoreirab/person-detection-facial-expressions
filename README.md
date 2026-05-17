<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Kaggle-GPU%20T4-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Fase%201%20Concluída-brightgreen?style=for-the-badge"/>

<br/><br/>

# 👁️ Visão Computacional — Reconhecimento de Pessoas

### Projeto acadêmico de visão computacional em 3 fases: classificação, detecção e reconhecimento de expressões faciais

**Universidade** · Grupo 05 · 2026

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
✅ Fase 1 — Reconhecimento de pessoas numa imagem          [CONCLUÍDA]
           Classificador binário com Transfer Learning
           (AlexNet · ResNet18 · EfficientNet-B0)

🔜 Fase 2 — Segmentação e detecção de pessoas             [Em breve]
           Detecção com bounding box usando YOLO ou DETR

🔜 Fase 3 — Reconhecimento de expressão facial            [Em breve]
           Classificação de emoções a partir dos rostos detectados
```

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

## 📁 Estrutura do Repositório

```
visao-computacional/
│
├── fase-1/                          # Fase 1: Classificação Binária
│   ├── alexnetexperiment.ipynb      # AlexNet — 3 regimes × 5 folds + Grad-CAM
│   ├── resnet18.ipynb               # ResNet18 — 3 regimes × 5 folds + Grad-CAM
│   ├── efficientnet-b0.ipynb        # EfficientNet-B0 — 3 regimes × 5 folds + Grad-CAM
│   └── minirelatorio.txt            # Mini-relatório da fase
│
├── fase-2/                          # 🔜 Fase 2: Detecção/Segmentação
│   └── (em desenvolvimento)
│
├── fase-3/                          # 🔜 Fase 3: Expressão Facial
│   └── (em desenvolvimento)
│
└── README.md
```

---

## 🔬 Análise Crítica

### Pontos Fortes
- **Protocolo rigoroso:** Validação cruzada estratificada com 5 folds garante estimativas robustas e sem vazamento de dados.
- **Três regimes de transfer learning** implementados e comparados de forma sistemática.
- **Diversidade arquitetural:** uma rede clássica (AlexNet), uma residual (ResNet18) e uma eficiente moderna (EfficientNet-B0) — cada uma com características distintas.
- **Grad-CAM** implementado para interpretabilidade, selecionando automaticamente o melhor fold.
- **Scheduler cosine annealing** no Full FT para estabilizar o treinamento em longa duração.

### Limitações e Próximos Passos
- Inferências incompletas em alguns cenários de borda — métricas por classe poderiam ser mais granularizadas no JSON de saída.
- Visualização de métricas pode ser aprimorada com gráficos interativos (ex: Plotly).
- Explorar **data augmentation mais agressivo** (mixup, cutout) para reduzir overfitting no Full FT.
- Considerar **ensemble** dos três modelos para ganho de performance adicional.

---

## 🛠️ Como Reproduzir

### Pré-requisitos

```bash
pip install torch torchvision timm torchmetrics scikit-learn matplotlib seaborn pandas numpy
```

### Dataset

1. Baixe o **Pascal VOC 2012** em: http://host.robots.ox.ac.uk/pascal/VOC
2. Extraia para `/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012` (ou ajuste o caminho no notebook)

### Execução

```bash
# Rodar cada notebook sequencialmente:
jupyter nbconvert --to notebook --execute alexnetexperiment.ipynb
jupyter nbconvert --to notebook --execute resnet18.ipynb
jupyter nbconvert --to notebook --execute efficientnet-b0.ipynb
```

> **Recomendado:** Executar no **Kaggle** com GPU T4 habilitada. Cada notebook leva ~2–3h com GPU.

---

## 📚 Referências

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Everingham, M., et al. (2010). *The Pascal Visual Object Classes (VOC) Challenge.* IJCV.

---

<div align="center">

**Fase 1 concluída em 08/05/2026**

*Projeto desenvolvido para a disciplina de Visão Computacional — 2026*

</div>
