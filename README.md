# Project Name

This repository is focused on determining the **thermal stability of DNA duplexes**, including:
- **Naturally occurring DNA duplexes**
- **DNA duplexes containing mismatches**
- **DNA duplexes with chemically modified nucleotides**, with modifications occurring on the sugar, base, or backbone of the DNA strand.

Key topics explored in this project include:
- **Feature Engineering**: DNA score, sequence encoding, and sequence statistics.
- **Machine Learning Models**: Multiple Linear Regression (MLR), Random Forest (RF), k-Nearest Neighbors (KNN), and deep learning models (e.g., CNNs).
- **Single vs. Multitask Learning**: Comparative analysis of single-task and multitask learning models for DNA duplex predictions.
- **Validation**: The models are validated on natural and chemically modified DNA duplexes, and the project aims to extend the dataset to include mismatched DNA sequences.

Additionally, the project involves stress testing the applicability of the machine learning models by using synthetic data, which explores the impact of DNA duplex conformations on the final predictions. There is ongoing interest in investigating **Molecular Dynamics (MD) simulations** to enhance the understanding of DNA duplex behavior.

> **Note**: This repository does not contain all the files from the development/research environment. However, key steps such as data exploration, experimental design, and visualizations are included.

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
      - [Use_of_Synthetic_data](#use_of_synthetic_data)
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
