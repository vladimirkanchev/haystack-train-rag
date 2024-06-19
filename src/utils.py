"""Construct utility functions to generate useful data."""
import pickle
from typing import List, Tuple


def create_gt_answer_data() -> List[str]:
    """Create utility ground truth data - answers for questions."""
    all_gt_answers = [
        'It was said to be 70 cubits (105 feet [32 metres]) tall, '
        + 'and it depicted the sun god Helios. Many representations '
        + 'of the statue depict the figure as nude or semi-nude, '
        + 'except for a cloak, '
        + 'and a representation in one relief suggests '
        + 'that the figure was shielding its eyes with one hand',
        'The Great Pyramid of Giza was the tomb of pharaoh Khufu, and '
        + 'still contains his granite sarcophagus. It had, '
        + 'like other tombs of Egyptian elites, four main purposes: '
        + 'It housed the body of the deceased and kept it safe. '
        + 'It demonstrated the status of the deceased and his family.',
        'It was 115 m (377 ft) long and 46 m (151 ft) wide, '
        + 'supposedly the first Greek temple built of marble. '
        + 'Its peripteral columns stood some 13 m (40 ft) high, '
        + 'in double rows that formed a wide ceremonial passage '
        + 'around the cella that housed the goddess\'s cult image.'
    ]

    return all_gt_answers


def create_question_data() -> List[str]:
    """Create utility question data for seven ancient world wonders."""
    all_questions = ['What does Rhodes Statue look like?',
                     'What was known for the The Great Pyramid of Giza, '
                     + 'Egypt in the Antiquity?',
                     'What does The Temple of Artemis at Ephesus look like?']

    return all_questions


def save_eval_data(rag_answers: List[str], retrieved_docs: List[List[str]]) \
        -> None:
    """Save evaluation data of the rag algo/given pipeline."""
    file_path = 'data/eval_data.pkl'
    with open(file_path, "wb") as file_in:
        # Writing data to a file
        pickle.dump([rag_answers, retrieved_docs], file_in)


def load_eval_data() -> Tuple[List[str], List[List[str]]]:
    """Load evaluation data of the rag algo/given pipeline."""
    file_path = 'data/eval_data.pkl'
    with open(file_path, "rb") as file_out:
        # Writing data to a file
        [rag_answers, retrieved_docs] = pickle.load(file_out)

    return rag_answers, retrieved_docs
