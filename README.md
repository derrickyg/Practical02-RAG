# Instructions to run
##### > pip install -r requirements-lock.txt (optional venv)
##### > docker-compose up -d (docker desktop open)
##### > install ollama
##### >>> ollama pull <model> (tinyllama and mistral)
##### > py experiments.py
## Compatible with python v3.11.2
# More about the project
#### We have default support for three text embedding models, two ollama LLMS, and three vectorDBs.
#### The best point of entry for this project is experiments.py, where we have listed all our tunable parameters. 
#### driver.py outlines the entire pipeline 
#### scripts/ directory includes preprocessing and running the llm locally
#### vector_store directory includes each DB we used
