<div align="center">
  <img src="/_media/seven_wonders.jpeg" width="800" height="500">
</div>

# Ask AI about Seven Wonders of the Ancient World

This repository contains a toy AI project that provides information about a given wonder of the ancient world using AI/LLM model and related wikipedia data. Our idea is to help aspiring historians and AI nerds to extend their knowledge about ancient history and get skills how to use AI/LLM models with RAG algorithm to solve similar problems using a large, publicly available, knowledge base.

## A List of Seven Ancient Wonders:
The seven ancient wonders are the architectural and artistic monuments in Eastern Mediterenean and Middle East during Antiquity which symbolize human ingenuity and architectural excellence. Unfortunately only the Great Pyramiad of Giza still exists today while others were destroyed during late Antiguity and our concept for them is based on historical documents and memories of people living from that epoch. 

List of Ancient Wonders include:

- Temple of Artemis
- Great Pyramid of Giza
- Hanging Gardens of Babylon
- Colossus of Rhodes
- Lighthouse of Alexandria 
- Mausoleum at Halicarnassus
- Statue of Zeus


Our Q&A system allows the user to formulate its own questions - 'What's the Colossus of Rhodes', ask the system, and obtain the answer.

<div align="center">
  <img src="/_media/seven_wonders_map.jpg" width="800" height="500">
</div>


## Technical details 

At the current moment the project represents Q&A system based on RAG algorithm using Haystack 2.0 framework. The backend is provided by FastAPI while the frontend represents a by simple javascript app at the moment. The knowledge base is stored in specialized (in-memory store at the moment) embedding store as vectors, which are used later to build the query context. The aim of intelligent part of the Q&A system is to find out the context of each query through distance calculation between the query embedding and all vectors. Finally the extended query (context + query) is sent to the LLM model and its response serve as a Q&A system answer.

Actually the Q&A system provides two functionalities: the first one for usual Q&A system where a user can ask questions and receive answers. The second one uses hard-coded questions with ground-truth answers for RAG algorithm evaluation and then can perform additional research and/or thus the user can extend it for other purposes.
    
It started as a training project based on a notebook [1] as a simple Q&A system to answer questions for seven wonders of the ancient world. The we wanted it to grow it into a professional project solving more complex problems with the rag algorithm while use real, unprocessed data.


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
2. Start your local environment (replace '.haystack-env' with your own environment):
```
source .haystack-env/bin/activate 
```
3. Before you start the Q&A program locally, run the script to convert PDF documents to vector embeddings and save them into the haystack storage:
```
python src/ingest.py
```

4. You have two options to consider: ask a question to the Q&A system (5), or run it see how it works and evaluation results (6).


5. Then run the following script to process inquiries about a certain ancient world wonder and to obtain rag evaluation:
```
python main.py
```

or

6. Run the following script to ask a simple question about one of the seven wonders from the ancient world:
'''
python app.py
'''

Thus will start the local fastapi server, call it through writing localhost:8801 and finally you can enter your question through a simple ui interface:

<div align="center">
  <img src="/_media/ui_fastapi_rag.jpg" width="700" height="450">
</div>

## Technologies

At the current moment we use the following software technologies:
    
- Visual Studio Code 1.90.0
- Python 3.10.12
- Docker 26.1

    
## Python Packages Used
    
Some of the python packages which are part of our project:

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

At the current moment, we have implemented three RAG algorithms: dense (sentence transformers, cos distance), sparse (bm25 algorithm, no-embedding), and hybrid (both dense and sparse) algorithms. We have used two types of LLM models: proprietary *openai gpt-3.5* and open-source *HuggingFaceH4/zephyr-7b-beta* model. We have already implemented two evaluation metrics for the RGA algorithms: *faithfulness* (measure factual consistency of the answer against the retrieved context - reverse of presence of hallucinations) and *sas_evaluator* (measure semantic similarity between the predicted answer and the ground-truth answer using a fine-tuned language model).

We plan to extend the evaluation part using metrics from *deep-eval* and *RAGAS* frameworks. The issue we have here is related to general lack of ground-truth data for questions.


## Future Work

Our next tasks are as follows:
   
- apply a preprocessing algorithm
- apply better and more accurate rag algorithms
- add a vector database instead of in-memory datastore 
- add another ui - streamlit
- extend the evaluation part with other metrics


## Who wants to contribute

Contributions, issues and feature requests will be welcomed at the later stage of project development. 


## Reference
[1] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline