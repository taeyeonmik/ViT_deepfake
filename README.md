# Détection de Deepfake

Ce projet utilise une combinaison de modèles EfficientNet et Vision Transformer pour détecter les deepfakes dans les images.

## Table des matières
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset]($dataset)
4. [Structure du Projet](#structure-du-projet)
5. [Modèles Utilisés](#modèles-utilisés)
6. [Présentation](#presentation)
7. [Résultats](#resultat)

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
ffhq.zip / stylegan_ffhq.zip/ stylegan_celeba.zip. Consultez [On the Detection of Digital Face Manipulation](https://arxiv.org/abs/1910.01717) 

## Structure du Projet
votre-projet/
├── README.md
├── requirements.txt
├── deepfake_detection/
│   ├── __init__.py
│   ├── efficientnet_model.py
│   ├── vision_transformer_model.py
│   ├── detection.py
├── examples/
│   ├── example_image.jpg
│   ├── example_video.mp4

## Modèles Utilisés
### EfficientNet
EfficientNet est une famille de modèles de convolution neuronale qui optimisent la précision et l'efficacité des calculs. Pour plus d'informations, consultez [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).

### Vision Transformer (ViT)
Vision Transformer (ViT) utilise des mécanismes d'attention pour traiter des images avec une précision impressionnante. Pour plus d'informations, consultez [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

## Présentation
Vous pouvez consulter `presentation.pdf` dans ce repo, si vous voulez savoir des détailles de ce projet.

## Résultats
### AUC
- EfficientNet-B0 (Baseline) = 0.880
- EfficientNet-B0 + ViT = 0.981
- EfficientNet-B1 + ViT = 0.984
