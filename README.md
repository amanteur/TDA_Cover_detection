# TDA_Cover_detection
---
This is a work code for the coursework "Application of Topological Data Analysis in Music Information Retrieval" \
in terms of Master Program "Data Science" in the first year of study.
---
This code uses as base dataset [covers80](https://labrosa.ee.columbia.edu/projects/coversongs/covers80/) dataset.

Here is a small description of each file:
1. In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_audio_preprocessing_0.py) .py file I create point clouds from .mp3 files from dataset
2. In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_CoverDetection_1_Data.ipynb) .ipynb notebook I place all data (point clouds) in DataFrames in order to access them easily.
3.  In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_CoverDetection_2_Persistence_Diagrams.ipynb) .ipynb notebook I create from point clouds persistence diagrams.
4.  In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_CoverDetection_Topological_Features_3.ipynb) .ipynb notebook I make Rips-filtration on point clouds and extract persistence diagrams from them, then extract topological features from persistence diagrams.
5.  In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_CoverDetection_4_ML_Models_Preprocessing.ipynb) .ipynb notebook I create paired dataset from feature datasets and split it test/train, with new mutual features.
6.  In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_CoverDetection_5_ML_Models_Evaluation.ipynb) .ipynb notebook I evaluate machine learning models on extracted features of mutual datasets.
7.  In [this](https://github.com/amanteur/TDA_Cover_detection/blob/main/TDA_CoverDetection_6_Siamese_Network.ipynb) .ipynb notebook I examine another approach for cover detection task, which is using Siamese Neural Network.

