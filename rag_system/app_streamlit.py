"""An entrypoint file streamlit gui of seven wonders app."""
import os
from pathlib import Path
import sys
from typing import Tuple

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
import streamlit as st

from rag_system.eval_pipelines import evaluate_gt_pipeline
from rag_system.inference import run_pipeline
from rag_system.rag_pipelines import select_rag_pipeline
from rag_system.utils import create_gt_data
from rag_system.utils import create_qui_question_data

NUM_COLS = 2
PARAMS = ["faithfulness: "]
VALS_STR = ['val1', 'val2']

def get_result_streamlit(query: str) -> Tuple[str, float]:
    """Run inference on the rag pipeline."""
    rag_pipeline = select_rag_pipeline()
    eval_pipeline = evaluate_gt_pipeline()
    rag_answer, retrieved_docs = run_pipeline(query, rag_pipeline)
    results = eval_pipeline.run({
        "faithfulness": {"questions": [query],
                         "contexts": [retrieved_docs],
                         "predicted_answers": [rag_answer]},
    }
    )

    return rag_answer, results['faithfulness']['score']


def initialize() -> None:
    """Initialize streamlit guis areas."""    
    if VALS_STR[0] not in st.session_state:
        st.session_state[VALS_STR[0]] = ""
    if VALS_STR[1] not in st.session_state:
        st.session_state[VALS_STR[1]] = ""
    if 'parm_text' not in st.session_state:
        st.session_state.parm_text = ""


def enter_wonder_question() -> None:
    """Generate and evauate AI answer for a question."""
    st.title("AI App for the Seven Ancient Wonders:")

    question_gui_data = create_qui_question_data()
    ground_truth_data = create_gt_data()
    initialize() 

    left_column, right_column = st.columns(NUM_COLS)
    with right_column:
        st.text_area("AI generated answer",
                     value=st.session_state[VALS_STR[0]],
                     height=200)
        st.text_area("Ground truth answer",
                     value=st.session_state[VALS_STR[1]],
                     height=200)
        st.write(st.session_state.parm_text)

    with left_column:
        query = st.selectbox(
            "Select a question about your wonder...",
            question_gui_data, index=None,
            placeholder="Select a question...",
        )
        st.write("You selected:", query)
        if st.button("Ask AI"):
            # Update the two text areas and parameter value with content
            rag_answer, param_value = get_result_streamlit(query)
            st.session_state[VALS_STR[0]] = rag_answer
            st.session_state[VALS_STR[1]] = ground_truth_data[query]
            st.session_state.parm_text = f"{PARAMS[0]}: {param_value}"
            st.rerun()


def run() -> None:
    """Run streamlit gui application for ai rag answering."""
    enter_wonder_question()


if __name__ == "__main__":
    run()
