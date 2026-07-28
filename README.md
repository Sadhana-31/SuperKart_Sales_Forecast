# 🛒 SuperKart Sales Forecaster

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff69b4.svg)](https://gradio.app/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![CI/CD](https://img.shields.io/badge/CI%2FCID-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

An end-to-end MLOps pipeline for predicting product store sales revenue. This project automates the complete machine learning lifecycle—from data ingestion and preprocessing to model training, evaluation, and deployment as an interactive web application.

## 🚀 Features

- **Automated CI/CD Pipeline:** Fully automated workflow using GitHub Actions.
- **High-Performance Model:** Tuned XGBoost Regressor for accurate sales predictions.
- **Interactive Dashboard:** Sleek, enterprise-grade UI built with Gradio.
- **Cloud Deployment:** Seamlessly hosted on Hugging Face Spaces (ZeroGPU tier).
- **REST API:** Programmatic access automatically generated via Gradio.

## 🏗️ Architecture & Pipeline

The pipeline is triggered automatically on every push to the `main` branch:

1. **Data Registration:** Raw data is uploaded and versioned on Hugging Face.
2. **Data Preparation:** Data is cleaned, missing values are imputed, and features are encoded.
3. **Model Training:** An XGBoost model is trained and hyperparameter-tuned.
4. **Deployment:** The model is bundled with the Gradio app and deployed to Hugging Face Spaces.

## 📊 Dataset & Model

- **Target Variable:** `Product_Store_Sales_Total` (INR)
- **Input Features:** Product Weight, Display Area, MRP, Sugar Content, Product Category, Store Age, Store Size, City Tier, and Store Type.
- **Algorithm:** XGBoost (Extreme Gradient Boosting)

## 💻 Live Demo

**Try the live application here:**  
🔗 [SuperKart Sales Forecaster on Hugging Face](https://huggingface.co/spaces/Sadhana3105/SuperKart-Sales-Forecast)

## 🛠️ Local Installation

To run this project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sadhana3105/SuperKart_Sales_Forecast.git
   cd SuperKart_Sales_Forecast
   ```

2. **Install dependencies:**
   ```bash
   pip install -r superkart_project/deployment/requirements.txt
   ```

3. **Run the application:**
   ```bash
   python superkart_project/deployment/app.py
   ```
   The app will be available at `http://localhost:7860`.

## 📂 Repository Structure

```text
SuperKart_Sales_Forecast/
├── .github/workflows/
│   └── pipeline.yml          # GitHub Actions CI/CD configuration
├── superkart_project/
│   ├── data/                 # Raw and processed datasets
│   ├── model_building/       # Scripts for training and data prep
│   ├── deployment/           # Gradio app.py and requirements
│   └── hosting/              # HF Spaces deployment scripts
└── README.md                 # Project documentation
```
