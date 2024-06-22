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

The user can formulate its question and ask the Q&A system.

<div align="center">
  <img src="/_media/seven_wonders_map.png" width="800" height="500">
</div>

## Technical details 
    At the current moment the project represents Q&A system based on RAG algorithm using Haystack 2.0 framework. The backend is provided by FastAPI and frontend by simple javascript app at the moment.Actually there are two functionality: the first one for usual Q&A system while the second one provides hard-coded questions with ground-truth answers for RAG algorithm evaluation.
    
    It started as a training project based on a notebook [1] as a simple Q&A system to answer questions for seven wonders of the ancient world. We wanted  to grow it into a professional project solving more complex problems with the rag algorithm and extend it to solve other problems with real, unprocessed data.

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
4. You have two options to consider: ask a question to the Q&A system, or run it see how it works and evaluation results.

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

- datasets==2.19
- fastapi == 0.111
- haystack-ai==2.2
- huggingface_hub==0.23
- openai==1.33
- pandas==2.2
- tokenizers==0.19
- torch==2.3
- transformers==4.41


## Results and Evaluations

At the current moment, we have implemented three RAG algorithms. dense (sentence transformers, cos dist), sparse (bm25), and hybrid (both dense and sparse) algorithms. We use two types of LLM models: proprietary openai gpt-3.5 and open-source 'HuggingFaceH4/zephyr-7b-beta' model. We implemented two evaluation metrics for the RGA algorithms: faithfulness (measure factual consistency of the answer against the retrieved context-presence of hallucinations) and sas_evaluator (measure semantic similarity betweeb tge predicted answer and the ground truth answer using a fine-tuned language model).

We plan to extend the evaluation part using more evaluation metrics from deep-eval and RAGAS frameworks. The issue we have here is related to general lack of ground-truth data for questions.


## Future Work

Our next tasks are as follows:
   
- apply a preprocessing algorithm
- apply better rag algorithms
- add a vector database
- add another ui - streamlit
- extend evaluation part


## Who wants to contribute

Contributions, issues and feature requests will be welcomed at the later stage of project development. 


## Reference
[1] https://haystack.deepset.ai/tutorials/27_first_rag_pipeline