# Détection de Deepfake : EfficientNet + Vision Transformer

Ce projet utilise une combinaison de modèles EfficientNet et Vision Transformer pour détecter les deepfakes dans les images.

## Table des matières
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset]($dataset)
4. [Modèles Utilisés](#modèles-utilisés)
5. [Présentation](#presentation)
6. [Résultats](#resultat)

## Introduction
La détection des deepfakes est un enjeu majeur de la sécurité numérique. Ce projet combine deux architectures de réseaux de neurones, EfficientNet et Vision Transformer, pour améliorer la précision et la robustesse de la détection de deepfakes.

## Installation
Pour utiliser ce projet, vous devez avoir Python 3.8 et les bibliothèques nécessaires installées. Voici comment configurer votre environnement :

```bash
git clone https://github.com/taeyeonmik/ViT_deepfake
cd ViT_deepfake
pip install -r requirements.txt
```

[Installation de Model State Dict](https://drive.google.com/drive/folders/1iqpvkxa0oxgub9URjgcH7Sco-TZzhiOP?usp=sharing)

## Dataset
### DFFD: Diverse Fake Face Dataset
`ffhq.zip`, `stylegan_ffhq.zip`, `stylegan_celeba.zip`. Consultez [On the Detection of Digital Face Manipulation](https://arxiv.org/abs/1910.01717) pour l'article, et [DFFD: Diverse Fake Face Dataset](https://cvlab.cse.msu.edu/dffd-dataset.html#bibtex-detection-of-digital-face-manipulation) pour les datasets.

## Modèles Utilisés
### EfficientNet
EfficientNet est une famille de modèles de convolution neuronale qui optimisent la précision et l'efficacité des calculs. Pour plus d'informations, consultez [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).

### Vision Transformer (ViT)
Vision Transformer (ViT) utilise des mécanismes d'attention pour traiter des images avec une précision impressionnante. Pour plus d'informations, consultez [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

## Présentation
Vous pouvez consulter `presentation.pdf` dans ce repo, si vous voulez savoir des détailles de ce projet.

## Résultats
Tous les résultats sont disponible en forme de graphiques dans `train-results` et `inference`.
### Validation (À 30ème epoch)
|                 | Accuracy        | Precision       | Recall          | F1              |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| EffNetB0        | 0.7499          | 0.7483          | 0.7528          | 0.7505          |
| EffNetB0 + ViT  | 0.8854          | 0.9355          | 0.8278          | 0.8784          |
| EffNetB1 + ViT  | 0.9040          | 0.9113          | 0.8949          | 0.9030          |

### Inference (AUC)
|                 | AUC             |
|-----------------|-----------------|
| EffNetB0        | 0.880           |
| EffNetB0 + ViT  | 0.981           |
| EffNetB1 + ViT  | 0.984           |


