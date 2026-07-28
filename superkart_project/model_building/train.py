import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow setup 
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("SuperKart-Sales-Forecast-Experiment")

# Hugging Face API 
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load data from Hugging Face 
base = "hf://datasets/Sadhana3105/superkart/"
Xtrain = pd.read_csv(base + 'Xtrain.csv')
Xtest  = pd.read_csv(base + 'Xtest.csv')
ytrain = pd.read_csv(base + 'ytrain.csv').squeeze()
ytest  = pd.read_csv(base + 'ytest.csv').squeeze()
print('Data loaded from Hugging Face.')

# Feature definitions 
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Age'
]
categorical_features = [
    'Product_Sugar_Content',
    'Product_Type',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]

# Preprocessing pipeline 
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Helper: compute regression metrics 
def evaluate(model, X, y, label=''):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae  = mean_absolute_error(y, preds)
    r2   = r2_score(y, preds)
    print(f"{label} -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return rmse, mae, r2

# Model 1: Decision Tree (baseline) 
print('\n Decision Tree Regressor ')
with mlflow.start_run(run_name='Decision_Tree'):
    dt_pipeline = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=42))
    dt_pipeline.fit(Xtrain, ytrain)
    tr_rmse, tr_mae, tr_r2 = evaluate(dt_pipeline, Xtrain, ytrain, 'Train')
    ts_rmse, ts_mae, ts_r2 = evaluate(dt_pipeline, Xtest,  ytest,  'Test ')
    mlflow.log_params({'model': 'DecisionTree', 'random_state': 42})
    mlflow.log_metrics({
        'train_rmse': tr_rmse, 'train_mae': tr_mae, 'train_r2': tr_r2,
        'test_rmse':  ts_rmse, 'test_mae':  ts_mae, 'test_r2':  ts_r2
    })

# Model 2: Random Forest 
print('\n Random Forest Regressor ')
with mlflow.start_run(run_name='Random_Forest'):
    rf_pipeline = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    rf_pipeline.fit(Xtrain, ytrain)
    tr_rmse, tr_mae, tr_r2 = evaluate(rf_pipeline, Xtrain, ytrain, 'Train')
    ts_rmse, ts_mae, ts_r2 = evaluate(rf_pipeline, Xtest,  ytest,  'Test ')
    mlflow.log_params({'model': 'RandomForest', 'n_estimators': 100, 'random_state': 42})
    mlflow.log_metrics({
        'train_rmse': tr_rmse, 'train_mae': tr_mae, 'train_r2': tr_r2,
        'test_rmse':  ts_rmse, 'test_mae':  ts_mae, 'test_r2':  ts_r2
    })

# Model 3: Gradient Boosting 
print('\n Gradient Boosting Regressor ')
with mlflow.start_run(run_name='Gradient_Boosting'):
    gb_pipeline = make_pipeline(preprocessor, GradientBoostingRegressor(n_estimators=100, random_state=42))
    gb_pipeline.fit(Xtrain, ytrain)
    tr_rmse, tr_mae, tr_r2 = evaluate(gb_pipeline, Xtrain, ytrain, 'Train')
    ts_rmse, ts_mae, ts_r2 = evaluate(gb_pipeline, Xtest,  ytest,  'Test ')
    mlflow.log_params({'model': 'GradientBoosting', 'n_estimators': 100, 'random_state': 42})
    mlflow.log_metrics({
        'train_rmse': tr_rmse, 'train_mae': tr_mae, 'train_r2': tr_r2,
        'test_rmse':  ts_rmse, 'test_mae':  ts_mae, 'test_r2':  ts_r2
    })

# Model 4: XGBoost with GridSearchCV (best model) 
print('\n XGBoost Regressor with GridSearchCV ')
xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)

param_grid = {
    'xgbregressor__n_estimators':    [50, 100, 150],
    'xgbregressor__max_depth':        [3, 4, 5],
    'xgbregressor__learning_rate':    [0.01, 0.05, 0.1],
    'xgbregressor__colsample_bytree': [0.5, 0.7, 1.0],
    'xgbregressor__reg_lambda':       [0.5, 1.0],
}

xgb_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(
    xgb_pipeline, param_grid,
    cv=5, scoring='neg_root_mean_squared_error',
    n_jobs=-1, verbose=1
)

with mlflow.start_run(run_name='XGBoost_GridSearchCV'):
    grid_search.fit(Xtrain, ytrain)
    best_model = grid_search.best_estimator_
    print(f'Best params: {grid_search.best_params_}')

    tr_rmse, tr_mae, tr_r2 = evaluate(best_model, Xtrain, ytrain, 'Train')
    ts_rmse, ts_mae, ts_r2 = evaluate(best_model, Xtest,  ytest,  'Test ')

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        'train_rmse': tr_rmse, 'train_mae': tr_mae, 'train_r2': tr_r2,
        'test_rmse':  ts_rmse, 'test_mae':  ts_mae, 'test_r2':  ts_r2
    })

    # Save model locally and log as artifact
    model_path = 'best_superkart_model_v1.joblib'
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path='model')
    print(f'Model saved: {model_path}')

# Register best model on Hugging Face Model Hub 
repo_id   = 'Sadhana3105/superkart-sales-model'
repo_type = 'model'
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj='best_superkart_model_v1.joblib',
    path_in_repo='best_superkart_model_v1.joblib',
    repo_id=repo_id,
    repo_type=repo_type,
)
print('Best XGBoost model uploaded to Hugging Face Model Hub.')
