# Polish Sentiment Analysis Classifier

![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A simple machine learning model to classify the sentiment of educational course reviews written in Polish. This project serves as a practical demonstration of NLP (Natural Language Processing) basics for a university assignment.

## Project Overview

The main goal of this project is to automatically classify text reviews as either **Positive (1)** or **Negative (0)**. The model is trained on a small, balanced dataset of 20 hand-crafted reviews about fictional educational courses.

### Features
-   Classification of text sentiment in Polish.
-   Uses the TF-IDF vectorization technique to convert text into numerical data.
-   Employs a Logistic Regression model for binary classification.
-   Provides a detailed evaluation of the model's performance using standard metrics like Accuracy and a Confusion Matrix.

## Technologies Used
-   **Python 3**
-   **Scikit-learn:** For machine learning models, vectorization, and metrics.
-   **NumPy:** For efficient numerical operations.

## Setup and Installation

Follow these steps to set up the project locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/Polish-Sentiment-Analysis-Classifier.git
    cd Polish-Sentiment-Analysis-Classifier
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the classification script and see the results, execute the following command in your terminal:

```bash
python sentiment_classifier.py
```

The script will output the model's accuracy and a detailed confusion matrix directly to the console.

## Results and Evaluation

The model was trained and evaluated on the same set of 20 reviews. On this dataset, it achieved **100% accuracy**.

### Confusion Matrix
A confusion matrix provides a more detailed breakdown of the model's performance.

|                      | Predicted: Negative | Predicted: Positive |
|----------------------|---------------------|---------------------|
| **Actual: Negative** | **TN = 10**         | FP = 0              |
| **Actual: Positive** | FN = 0              | **TP = 10**         |

-   **True Positives (TP):** 10 - Correctly identified positive reviews.
-   **True Negatives (TN):** 10 - Correctly identified negative reviews.
-   **False Positives (FP):** 0 - No negative reviews were misclassified as positive.
-   **False Negatives (FN):** 0 - No positive reviews were misclassified as negative.

The perfect score indicates that the model was able to find clear, distinguishable patterns in the provided text data.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.