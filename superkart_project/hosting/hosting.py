from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id   = 'Sadhana3105/SuperKart-Sales-Forecast'
repo_type = 'space'

# Create the Space as Static SDK (free, always works)
# The README.md we upload will tell HF to treat it as a Gradio Space
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk='static',
        private=False
    )
    print(f"Space '{repo_id}' created.")

# Push all deployment files (app.py, requirements.txt, Dockerfile, README.md)
api.upload_folder(
    folder_path='superkart_project/deployment',
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo='',
)
print('All deployment files pushed to Hugging Face Space.')
print('HF will detect the README.md SDK config and rebuild as a Gradio Space.')
