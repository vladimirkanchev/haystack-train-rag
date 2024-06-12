# haystack-train-rag

A toy AI project that uses a RAG algorithm with a haystack framework. It started as a training project based on a notebook [1] as a simple Q&A system to answer questions for seven wonders of the ancient world. Author's purpose was to grow it into a professional project with mlops, test functionality, dockerizing, web/ui interface, etc solving more complex problems with the rag algorithm.


## Requirements

To run this project, you will need a OPENAI API KEY token, which should be set in your local environment as follows:
```
OPENAI_API_KEY=''
```


## Installation, Setup and Run

1. To run the project in your local system, you need to clone and install the project first:
```
git clone https://github.com/vladimirkanchev/haystack-train-rag
cd haystack-train-rag
pip install -r requirements.txt
```
2. Before you start the Q&A program locally, run the script to convert PDF documents to vector embeddings and save them into the haystack storage:
```
python src/ingest.py
```
3. Then run the following script to process inquiries about a certain ancient world wonder and fetch the answer:
```
python main.py "What does Rhodes Statue look like?"
```

## Technologies

At the current moment we use the following software technologies:
    
- Visual Studio Code 1.90.0
- Python 3.10.12

    
## Python Packages Used
    
Some of the python packages which are part of our project:

- haystack-ai==2.2.1
- datasets==2.19.2
- openai==1.33.0
- pandas==2.2.2
- pillow==10.3.0
- pydantic==2.7.3
- tokenizers==0.19.1
- torch==2.3.0
- tqdm==4.66.4
- transformers==4.41.2


## Results and Evaluations

At the current moment, we have implemented only a simple RAG algorithm. We are also aware we need a solid test algorithm.


## Future Work

Our next tasks are as follows:
   
- apply own preprocessing algorithm
- apply dockerization
- add ui - streamlit


## Who wants to contribute

Contributions, issues and feature requests will be welcomed at the later stage of project development. 


## Reference
[1] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline