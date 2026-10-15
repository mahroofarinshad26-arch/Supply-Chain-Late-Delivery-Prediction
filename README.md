# Supply Chain Late Delivery Risk Prediction Using Deep Learning

## Project Overview

This project uses Deep Learning to predict late delivery risk in supply chain operations. It uses the DataCo Smart Supply Chain dataset to analyze order details, customer information, product details, and shipping information.

A Transformer-based model is developed to classify orders into two categories: late delivery risk and not late delivery risk.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Streamlit
* Matplotlib
* Seaborn

## Dataset

The project uses the DataCo Smart Supply Chain Dataset, which contains order, customer, product, and shipping-related information.

Dataset source: [DataCo Smart Supply Chain Dataset on Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

## Machine Learning / Deep Learning Model

A Transformer-based deep learning model is used to predict delivery risk based on categorical and numerical features.

## Project Features

* Supply chain data preprocessing
* Categorical feature encoding
* Numerical feature scaling
* Transformer-based deep learning model
* Late delivery risk prediction
* Model evaluation using accuracy, precision, recall, and F1-score
* Interactive Streamlit web application

## Model Performance

* Accuracy: 70.95%
* Precision: 81.92%
* Recall: 60.33%
* F1-score: 69.49%

## Project Files

* `app.py` – Streamlit application
* `supply_chain_transformer.pth` – Trained Transformer model
* `supply_chain_preprocessing.pkl` – Preprocessing objects
* `requirements.txt` – Required Python libraries

## How to Run the Project

1. Clone or download this repository.

2. Install the required libraries:

   `pip install -r requirements.txt`

3. Run the Streamlit application:

   `streamlit run app.py`

## Conclusion

This project demonstrates how Deep Learning can be used to predict late delivery risk in supply chain operations. The Transformer-based model analyzes order-related features and provides delivery risk predictions through an interactive Streamlit application. This project helps understand the role of data analysis and deep learning in improving supply chain management.


## Deployment

The application is intended to be deployed using Streamlit Community Cloud.

