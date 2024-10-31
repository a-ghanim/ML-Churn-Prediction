# ML Customer Churn Prediction

End-to-end pipeline for a customer churn prediction model, served for inference through a Streamlit app.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/a-ghanim/ml-customer-churn-prediction.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ml-customer-churn-prediction
    ```
3. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Jupyter Notebook to train the model:
    ```sh
    jupyter notebook
    ```
    Open the `notebooks/your_notebook.ipynb` file and run all cells to train the model.

2. Serve the model using Streamlit:
    ```sh
    streamlit run app.py
    ```

## Features

- Data preprocessing and feature engineering
- Model training and evaluation
- Model deployment using Streamlit
- Interactive user interface for predictions
- Pre-trained models included as pickle files

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```sh
    git checkout -b feature/YourFeature
    ```
3. Commit your changes:
    ```sh
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```sh
    git push origin feature/YourFeature
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
