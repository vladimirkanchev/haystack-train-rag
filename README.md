<div align="center">
  <img src="/_media/seven_wonders.png" width="800" height="500">
</div>

# Ask AI about Seven Wonders of the Ancient World

This repository contains a toy AI project that provides information about a certain wonder of the ancient world using LLM model trained on related wikipedia data. Our idea is aspiring historians and AI nerds to extend their knowledge about ancient history and how to use LLM models with RAG algorithm to solve using a wide knowledge base.

## A List of Seven Ancient Wonders:
The seven ancient wonders are the architectural and artistic monuments in Eastern Mediterenean and Middle East during Antiquity and are considered as perfect by the Greeks and the Romans. They also symbolize human ingenuity and architectural excellence but unfortunately only the Great Pyramiad of Giza still exists today. 
Wonders include:

- Great Pyramid of Giza
- Hanging Gardens of Babylon
- Temple of Artemis
- Statue of Zeus
- Mausoleum at Halicarnassus
- Colossus of Rhodes
- Lighthouse of Alexandria 


<div align="center">
  <img src="/_media/seven_wonders_map.png" width="800" height="500">
</div>

## Technical details 
    At the current moment the project represents Q&A system based on RAG algorithm using Haystack 2.0 framework. The backend is provided by FastAPI and frontend by simple javascript app at the moment.Actually there are two functionality: the first one for usual Q&A system while the second one provides ground truth for RAG algorithm evaluation. It started as a training project based on a notebook [1] as a simple Q&A system to answer questions for seven wonders of the ancient world. We wanted  to grow it into a professional project solving more complex problems with the rag algorithm and extend it to solve other problems with real, unprocessed data.

## Requirements

To run the AI models in the project, you will need a OPENAI API KEY token for the commercial OpenAI model or HF_API_TOKEN token for the open source AI model. They should be set in your local environment as follows:
```
OPENAI_API_KEY=''
```
```
HF_API_TOKEN=''
```

## Installation, Setup and Run

1. To run the project in your local system, you need to clone and install the project first:
```
git clone https://github.com/vladimirkanchev/haystack-train-rag
cd haystack-train-rag
pip install -r requirements.txt
```
2. Start your local environment:
```
source .haystack-env/bin/activate 
```
3. Before you start the Q&A program locally, run the script to convert PDF documents to vector embeddings and save them into the haystack storage:
```
python src/ingest.py
```
4.1 Then run the following script to process inquiries about a certain ancient world wonder and to obtain rag evaluation:
```
python main.py
```
or 
4.2 Run the following script to ask a simple question about one of the seven wonders from the ancient world:
'''
python app.py
'''
Thus will start local fastapi server and you can enter your question through a simple ui interface:

<div align="center">
  <img src="/_media/ui_fastapi_rag.jpg" width="800" height="500">
</div>

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
- fastapi == 0.111
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