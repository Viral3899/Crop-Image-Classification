import os
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO,format = '[%(asctime)s] : %(message)s')

project_name = 'Crop_Image_Classification'

list_of_files = [
     ".github/workflows/.gitkeep",
     f"src/{project_name}/__init__.py",
     f"src/{project_name}/components/__init__.py",
     f"src/{project_name}/utils/__init__.py",
     f"src/{project_name}/constants/__init__.py",
     f"src/{project_name}/config/__init__.py",
     f"src/{project_name}/config/configuration.py",
     f"src/{project_name}/entity/__init__.py",
     f"src/{project_name}/pipeline/__init__.py",
     'config/config.yaml',
     'params.yaml',
     'requirements.txt',
     'setup.py',    
     'research/trails.ipynb',
     'main.py',
     'app.py',
     'templates/index.html',
     'DockerFile',
     '.dockerignore'
     ]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Created directory : {filedir} for this : {filename}")
        
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)== 0):
        with open(filepath,"w") as f:
            pass
        
        logging.info(f"Created empty file for : {filepath}")
    else:
        
        logging.info(f"File already exists for : {filepath}")