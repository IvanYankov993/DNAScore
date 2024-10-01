# Project Name

This repository is dedicated to the development and experimentation of various machine learning (ML) models, focusing on single and multi-task convolutional neural networks (CNNs), data exploration, and model evaluation. It is structured to support ease of use, collaboration, and future scalability. The repository contains key modules and scripts to facilitate the training, evaluation, and deployment of machine learning models.

## Table of Contents
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Project Overview](#project-overview)
- [Folders Overview](#folders-overview)
  - [notebooks](#notebooks)
  - [cnn_single_multi_task_models](#cnn_single_multi_task_models) *(Under Refactoring)*
  - [experiments](#experiments)
  - [SK-learn](#sk-learn)
  - [exploration](#exploration)
  - [src](#src)
    - [experiments](#experiments-1)
    - [models](#models)
    - [parser](#parser)
    - [training](#training)
    - [utils](#utils)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

```plaintext
notebooks/
cnn_single_multi_task_models/  <-- Under Refactoring
experiments/
SK-learn/
    Use_of_Synthetic_data/
exploration/
    Explore_LZ_MH.ipynb
    Features_LZ_MH.ipynb
    ML_Experiments_LZ_MH copy.ipynb
    NN_method.ipynb
src/
    experiments/
    models/
        __init__.py
        model.py
        model_functions.py
        model_single_task.py
        modeldev.py
    parser/
        __init__.py
        parser.py
        parserw3DNApdb.py
    training/
        evaluate.py
        train_single_task.py
    utils/
        __init__.py
        utils.py
