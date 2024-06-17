"""Construct utility functions to generate useful data."""
from haystack import Document


def create_gt_answer_data():
    """Create utility ground truth data - answers for questions."""
    all_gt_answers = [
        Document('It was said to be 70 cubits (105 feet [32 metres]) tall, '
                 + 'and it depicted the sun god Helios. Many representations '
                 + 'of the statue depict the figure as nude or semi-nude, '
                 + 'except for a cloak, '
                 + 'and a representation in one relief suggests '
                 + 'that the figure was shielding its eyes with one hand')
    ]
    return all_gt_answers


def create_question_data():
    """Create utility question data for seven ancient world wonders."""
    all_questions = [
        ['What does Rhodes Statue look like?']
    ]

    return all_questions
