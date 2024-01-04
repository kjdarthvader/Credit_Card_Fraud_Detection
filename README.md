# Credit_Card_Fraud_Detection

## Introduction
In the intricate domain of financial security, the accurate classification of credit card transactions as fraudulent or legitimate stands as a paramount challenge, particularly given the highly imbalanced nature of transaction datasets. This Credit Card Fraud Detection project harnesses the power of advanced machine learning techniques to effectively distinguish between these two classes of transactions. Confronting a dataset where fraudulent activities constitute a mere 492 out of 284,807 total transactions, the project is finely tuned to address this imbalance. Central to my strategy is the dual objective of minimizing false negatives, thereby avoiding the oversight of fraudulent activities, while concurrently keeping false positives to a minimum to prevent the undue flagging of legitimate transactions. Emphasizing precision and recall, the model is meticulously engineered to optimize the recall scores and the area under the Receiver Operating Characteristic (ROC) curve. These metrics are pivotal in evaluating the model's accuracy and reliability, ensuring a robust and effective tool in the battle against credit card fraud. This project not only demonstrates a profound understanding of the challenges inherent in fraud detection but also exemplifies a sophisticated approach to overcoming these hurdles, setting a new benchmark in the field of financial data analysis.

## Features 
- Customized Decision Boundary Optimization: Employs unique functions for optimizing decision boundaries in logistic regression and support vector machines, ensuring optimal model sensitivity and specificity.
- Ensemble Approach: Utilizes a voting system among various models including k-Nearest Neighbors, Support Vector Machines (with RBF and polynomial kernels), Logistic Regression, and Random Forest, to enhance detection accuracy.
- Robust Evaluation Metrics: Adopts comprehensive metrics such as recall and AUC for evaluating model performance, crucial in the context of imbalanced datasets.

## Technology Stack 
- Programming Language: Python 3.12.1
- Key Libraries: pandas, matplotlib, numpy, scikit-learn, scipy
- Dataset: Credit card transaction data, sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download.

## Getting Started

### Prerequisites
- Python >= 3.6.4
- Required Python Libraries: pandas, matplotlib, numpy, sklearn
### Installation Guide
1. Clone the Repository - git clone [repository-url]
2. Install Dependencies - pip install pandas matplotlib numpy sklearn
3. How to Run - python my_fraud_detection_model.py

