"""Construct utility functions to generate useful data."""
from typing import List


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
