import pytest
from pydantic import ValidationError

from polis.survey import SurveyAnswer, SurveyQuestion


def test_question_requires_at_least_two_options():
    with pytest.raises(ValidationError):
        SurveyQuestion(text="DST?", options=["only one"])


def test_answer_fields():
    a = SurveyAnswer(choice="Permanent DST", reason="more evening light")
    assert a.choice == "Permanent DST"
    assert a.reason
