"""An entrypoint file streamlit gui of seven wonders app."""
import os
import sys
# Add the parent directory to the sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(parent_dir, os.pardir)))

import streamlit as st


from eval_pipelines import evaluate_gt_pipeline
from inference import run_pipeline
from rag_pipelines import select_rag_pipeline
from utils import create_gt_data
from utils import create_qui_question_data


def get_result(query: str):
    """Run inference on the rag pipeline."""
    rag_pipeline = select_rag_pipeline()
    eval_pipeline = evaluate_gt_pipeline()
    rag_answer, retrieved_docs = run_pipeline(query[0], rag_pipeline)

    results = eval_pipeline.run({
        "faithfulness": {"questions": [query],
                         "contexts": [retrieved_docs],
                         "predicted_answers": [rag_answer]},
    }
    )

    return rag_answer, results['faithfulness']['score']


def enter_wonder_question() -> None:
    """Generate and evauate AI answer for a question."""
    st.title("AI App for the Seven Ancient Wonders:")

    question_gui_data = create_qui_question_data()
    ground_truth_data = create_gt_data()
    num_cols = 2

    left_column, right_column = st.columns(num_cols)
    if "val1" not in st.session_state:
        st.session_state["val1"] = ""
    if "val2" not in st.session_state:
        st.session_state["val2"] = ""
    if 'parm_text' not in st.session_state:
        st.session_state.parm_text = "faithfulness: "

    with right_column:
        st.text_area("AI generated answer", value=st.session_state["val1"],
                     height=200)
        st.text_area("Ground truth answer", value=st.session_state["val2"],
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
            # Update the two text area with content
            rag_answer, faithfulness_value = get_result(query)
            st.session_state["val1"] = rag_answer
            st.session_state["val2"] = ground_truth_data[query]
            # Update the session state for the parameter text
            st.session_state.parm_text = f"faithfulness: {faithfulness_value}"
            st.rerun()


def run() -> None:
    """Run streamlit gui application for ai rag answering."""
    enter_wonder_question()


if __name__ == "__main__":
    run()
