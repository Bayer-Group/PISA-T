# PISA-T

Prediction of Interference with Specific Assay - Technologies is a project aimed at developing and evaluating machine learning models for predicting assay interference based on statistically derived labels from ultra-large bioactivity data matrices. This repository contains scripts for data preprocessing, model training, validation, and testing.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/PISA-T.git
    ```

2. Install dependencies using Conda:

    ```bash
    conda env create -f environment.yml
    ```

## Usage

### 1. Data Preparation

- Place your raw data files in the `data/raw/` directory.
- Run the data preprocessing scripts in the `preprocessing/` directory to clean and preprocess the data.

### 2. Model Training

- Use scripts in the `dense_network/` and `random_forest/` directories to train different models.
- Modify hyperparameters and configurations as needed.

### 3. Model Validation

- Validate the models using validation scripts provided in the respective directories.
- Tune hyperparameters for optimal performance.

### 4. Testing

- Test the trained models on test data using testing scripts.
- Evaluate model performance and generate results.

## Contributors

- Vincenzo Palmacci (@vincenzo-palmacci)

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
