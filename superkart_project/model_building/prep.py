import pandas as pd
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

# ─── Load dataset from Hugging Face ───────────────────────────────────────
DATASET_PATH = "hf://datasets/Sadhana3105/superkart/SuperKart.csv"
df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─── Step 1: Drop ID columns ──────────────────────────────────────────────
# Product_Id and Store_Id are unique identifiers with no predictive signal
df.drop(columns=['Product_Id', 'Store_Id'], inplace=True)
print("Dropped ID columns: Product_Id, Store_Id")

# ─── Step 2: Standardise Product_Sugar_Content labels ─────────────────────
# Raw data contains dirty values such as 'reg', 'low fat', 'LF' etc.
sugar_map = {
    'low fat': 'Low Sugar',
    'LF':      'Low Sugar',
    'low sugar': 'Low Sugar',
    'Low Sugar': 'Low Sugar',
    'reg':     'Regular',
    'regular': 'Regular',
    'Regular': 'Regular',
    'no sugar': 'No Sugar',
    'No Sugar': 'No Sugar',
}
df['Product_Sugar_Content'] = df['Product_Sugar_Content'].str.strip().map(sugar_map)
print("Product_Sugar_Content standardised. Unique values:", df['Product_Sugar_Content'].unique())

# ─── Step 3: Impute missing values ────────────────────────────────────────
# Product_Weight: fill missing with column mean
weight_mean = df['Product_Weight'].mean()
df['Product_Weight'].fillna(weight_mean, inplace=True)

# Store_Size: fill missing with column mode
size_mode = df['Store_Size'].mode()[0]
df['Store_Size'].fillna(size_mode, inplace=True)
print("Missing values imputed: Product_Weight (mean), Store_Size (mode)")

# ─── Step 4: Feature Engineering ──────────────────────────────────────────
# Derive Store_Age from the establishment year for a more interpretable feature
df['Store_Age'] = 2025 - df['Store_Establishment_Year']
df.drop(columns=['Store_Establishment_Year'], inplace=True)
print("Feature engineered: Store_Age (2025 - Store_Establishment_Year)")

# ─── Step 5: Define features and target ───────────────────────────────────
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
target = 'Product_Store_Sales_Total'

X = df[numeric_features + categorical_features]
y = df[target]

# ─── Step 6: Train / Test split (80 / 20) ─────────────────────────────────
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print(f"Train size: {len(Xtrain)}, Test size: {len(Xtest)}")

# ─── Step 7: Save splits locally ──────────────────────────────────────────
Xtrain.to_csv('Xtrain.csv', index=False)
Xtest.to_csv('Xtest.csv',  index=False)
ytrain.to_csv('ytrain.csv', index=False)
ytest.to_csv('ytest.csv',  index=False)
print("Train/test splits saved locally.")

# ─── Step 8: Upload splits back to Hugging Face ───────────────────────────
files = ['Xtrain.csv', 'Xtest.csv', 'ytrain.csv', 'ytest.csv']
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id='Sadhana3105/superkart',
        repo_type='dataset',
    )
    print(f"Uploaded: {file_path}")
print("All train/test splits uploaded to Hugging Face.")
