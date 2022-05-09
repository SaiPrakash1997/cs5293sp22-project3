import pandas
from project3 import read_tsv, extract_sentences, normalize_sentences, model_definition
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def test_model_definition():
    file_name = "unredactor_test.tsv"
    dataFrame = read_tsv(file_name)
    names, test_names, temp_redacted_sentences, temp_test_redacted_sentences = extract_sentences(dataFrame)
    _sentences, _test_sentences = normalize_sentences(temp_redacted_sentences, temp_test_redacted_sentences)
    accuracy_score, precisionScore, recallScore, f1Score = model_definition(names, test_names, _sentences, _test_sentences)
    assert accuracy_score is not None
    assert precisionScore is not None
    assert recallScore is not None
    assert f1Score is not None
    assert test_names is not None
    assert type(_sentences) == list
    assert _sentences is not None
    assert len(_sentences) > 0
    assert type(_test_sentences) == list
    assert _test_sentences is not None
    assert len(_test_sentences) > 0
    assert names is not None
    assert type(temp_redacted_sentences) == list
    assert type(temp_test_redacted_sentences) == list
    assert type(names) == list
    assert type(test_names) == list
    assert temp_redacted_sentences is not None
    assert temp_test_redacted_sentences is not None
    assert type(dataFrame) == pandas.core.frame.DataFrame
    assert len(dataFrame) > 0




