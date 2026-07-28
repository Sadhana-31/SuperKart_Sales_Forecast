from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = 'Sadhana3105/superkart'
repo_type = "dataset"

# Authenticate with Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the dataset repo exists; create if it does not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created successfully.")

# Upload the entire data folder to Hugging Face
api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("SuperKart.csv uploaded to Hugging Face dataset repo successfully.")
