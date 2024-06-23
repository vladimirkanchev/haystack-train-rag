<div align="center">
  <img src="/_media/seven_wonders.jpeg" width="800" height="500">
</div>

# Ask AI about the Seven Wonders of the Ancient World

This repository contains a toy AI project that provides information about a given wonder of the ancient world using AI/a LLM model and related wikipedia data. Our idea is to help aspiring historians and AI nerds to extend their knowledge about ancient history and acquire skills to use AI/LLM models with a retrieval-augmented generation(RAG) algorithm to solve similar problems using a large knowledge base.

## A List of the Seven Ancient Wonders:
The seven ancient wonders are architectural and artistic monuments in the Eastern Mediterranean and the Middle East during antiquity, which symbolize human ingenuity and architectural excellence. Unfortunately, only the Great Pyramid of Giza still exists today, while others were destroyed during late antiguity and our concept of them is based on the historical documents and memories of people living in that age. 

The list of ancient wonders includes:

- Temple of Artemis
- Great Pyramid of Giza
- Hanging Gardens of Babylon
- Colossus of Rhodes
- Lighthouse of Alexandria 
- Mausoleum at Halicarnassus
- Statue of Zeus


Our Q&A system allows the user to formulate their own questions, e.g. to ask the system: "What's the Colossus of Rhodes?", ask the system, and obtain the answer.

<div align="center">
  <img src="/_media/seven_wonders_map.jpg" width="800" height="500">
</div>


## Technical details 

At the current moment, the project represents a Q&A system based on a RAG algorithm using a Haystack 2.0 framework. The backend is provided by FastAPI, while the frontend represents a simple javascript page at the moment. The knowledge base is stored in a specialized (in-memory store at the moment) store as embedding vectors, which are used later to build the query context. The aim of the intelligent part of the Q&A system is to find out the best context of each query through distance calculation between the query embedding and all embedding vectors. Finally, the extended query (context + query) is sent to the LLM model in the system and its response serves as a Q&A system answer.

Actually, our Q&A system provides two functionalities: the first functionality is that of an ordinary Q&A system, where a user can ask questions and receive answers. The second functionality uses hard-coded questions with ground-truth answers for RAG algorithm evaluation and then, users can perform additional research and can extend it for other purposes.
    
This project started as a training project based on a notebook [1] to answer questions about the seven wonders of the ancient world. Then, we wanted it to develop it into an AI system with the RAG algorithm.


## Requirements

To run the AI models in the project, you will need an OPENAI API KEY token for the commercial OpenAI model or an HF_API_TOKEN token for the open source AI model. They should be set in your local environment as follows:
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
2. Start your local environment (replace '.haystack-env' with your own environment):
```
source .haystack-env/bin/activate 
```
3. Before you start the Q&A program locally, run the script to convert PDF documents to vector embeddings and save them into the haystack storage:
```
python src/ingest.py
```

4. You have two options to consider: ask the Q&A system a question (5), or run it to see how it works and the evaluation results (6).


5. Then run the following script to process inquiries about a certain ancient world wonder and to obtain rag evaluation:
```
python main.py
```

or

6. Run the following script to ask a simple question about one of the seven wonders from the ancient world:
'''
python app.py
'''

This will start the local fastapi server, which you can access it through writing localhost:8001 and finally, you can enter your question through a simple user interface (UI):

<div align="center">
  <img src="/_media/ui_fastapi_rag.jpg" width="700" height="450">
</div>

## Technologies

In this project we use the following software technologies:
    
- Visual Studio Code 1.90.0
- Python 3.10.12
- Docker 26.1

    
## Python Packages Used
    
Some of the python packages which are used for our project are the following:

- datasets 2.19
- fastapi 0.111
- haystack-ai 2.2
- huggingface_hub 0.23
- openai 1.33
- pandas 2.2
- sentence-transformers 2.3
- tokenizers 0.19
- torch 2.3
- transformers 4.41


## Results and Evaluations

At the current moment, we have implemented three RAG algorithms: dense (sentence transformers, *cos* distance), sparse (bm25 algorithm, no-embedding), and hybrid (both dense and sparse) algorithms. We have used two types of LLM models: a proprietary *openai gpt-3.5* and an open-source *HuggingFaceH4/zephyr-7b-beta* model. We have already implemented two evaluation metrics for the RAG algorithms: *faithfulness* (measures the factual consistency of the answer against the retrieved context - reverse of presence of hallucinations) and *sas_evaluator* (measures the semantic similarity between the predicted answer and the ground-truth answer using a fine-tuned language model).

We plan to extend the evaluation part using metrics from *deep-eval*[2] and *RAGAS*[3] frameworks. The issue we have here is related to the general lack of ground-truth data for questions.


## Future Work

Our next tasks are as follows:
   
- apply a preprocessing algorithm
- apply better and more accurate RAG algorithms
- add a vector database instead of an in-memory datastore 
- add another UI - streamlit
- extend the evaluation part with other metrics


## Who wants to contribute

Contributions, issues and feature requests will be welcomed at a later stage of the project development. 


## Reference
[1] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline
[2] https://docs.confident-ai.com/docs/guides-rag-evaluation
[3] https://docs.ragas.io/en/latest/index.html