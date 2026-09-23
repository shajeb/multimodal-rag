import pytest
from rag.guardrails import citation_labels


@pytest.mark.parametrize("answer,expected", [
    ("North is largest [S1, S2].", {"1", "2"}),
    ("Claim [S1]. Other claim [S3].", {"1", "3"}),
    ("Unknown reference [S1, S99]", {"1", "99"}),
    ("A sentence without references.", set()),
])
def test_citation_labels(answer, expected):
    assert citation_labels(answer) == expected
