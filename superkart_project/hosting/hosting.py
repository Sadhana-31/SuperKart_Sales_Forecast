from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id   = 'Sadhana3105/SuperKart-Sales-Forecast'
repo_type = 'space'

# Create the Hugging Face Space if it does not already exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk='streamlit',
        private=False
    )
    print(f"Space '{repo_id}' created with Docker SDK.")

# Push all deployment files to the Space
api.upload_folder(
    folder_path='superkart_project/deployment',
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo='',
)
print('Deployment files pushed to Hugging Face Space. App is live!')
