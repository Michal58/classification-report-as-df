import numpy as np
import pytest
from bs4 import BeautifulSoup
import re
from classification_report_as_df import classification_report_as_df

DIFFERENCE_TOLERANCE = 1e-6

def test_headers_and_index():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    expected_columns = ['precision', 'recall', 'f1-score', 'support']
    assert df.columns.tolist() == expected_columns

    expected_index = ['0', '1', 'accuracy', 'micro', 'macro']
    assert df.index.tolist() == expected_index

def test_empty_input_raises_value_error():
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError) as excinfo:
        classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    assert "Found empty input array" in str(excinfo.value)

def test_one_class_classification():
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    assert df.shape == (4, 4)

    assert np.array_equal(
        np.array([
            [1.0, 1.0, 1.0, 5],
            ['', '', 1.0, 5],
            [1.0, 1.0, 1.0, 5],
            [1.0, 1.0, 1.0, 5]], 
            dtype=object
        ),
        df.to_numpy()
    )

def test_binary_classification():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1])

    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    zero_class_precision = df.loc['0', 'precision']
    zero_class_recall = df.loc['0', 'recall']
    zero_class_f1 = df.loc['0', 'f1-score']
    zero_class_support = df.loc['0', 'support']

    one_class_precision = df.loc['1', 'precision']
    one_class_recall = df.loc['1', 'recall']
    one_class_f1 = df.loc['1', 'f1-score']
    one_class_support = df.loc['1', 'support']

    accuracy_score = df.loc['accuracy', 'f1-score']
    accuracy_support = df.loc['accuracy', 'support']

    micro_class_precision = df.loc['micro', 'precision']
    micro_class_recall = df.loc['micro', 'recall']
    micro_class_f1 = df.loc['micro', 'f1-score']
    micro_class_support = df.loc['micro', 'support']

    macro_class_precision = df.loc['macro', 'precision']
    macro_class_recall = df.loc['macro', 'recall']
    macro_class_f1 = df.loc['macro', 'f1-score']
    macro_class_support = df.loc['macro', 'support']

    TARGET_ZERO_CLASS_F1 = 0.5714285714285714
    TARGET_ONE_CLASS_F1 = 0.7692307692307693

    assert np.isclose(zero_class_precision, 2/3, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(zero_class_recall, 2/4, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(zero_class_f1, TARGET_ZERO_CLASS_F1, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(zero_class_support, 4, atol=DIFFERENCE_TOLERANCE)

    assert np.isclose(one_class_precision, 5/7, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(one_class_recall, 5/6, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(one_class_f1, TARGET_ONE_CLASS_F1, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(one_class_support, 6, atol=DIFFERENCE_TOLERANCE)

    assert np.isclose(accuracy_score, 7/10, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(accuracy_support, 10, atol=DIFFERENCE_TOLERANCE)

    assert np.isclose(micro_class_precision, 7/10, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(micro_class_recall, 7/10, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(micro_class_f1, 7/10, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(micro_class_support, 10, atol=DIFFERENCE_TOLERANCE)

    assert np.isclose(macro_class_precision, (2/3 + 5/7)/2, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(macro_class_recall, (2/4 + 5/6)/2, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(macro_class_f1, (TARGET_ZERO_CLASS_F1 + TARGET_ONE_CLASS_F1)/2, atol=DIFFERENCE_TOLERANCE)
    assert np.isclose(macro_class_support, 10, atol=DIFFERENCE_TOLERANCE)

def test_multiclass_classification():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 2, 2, 2])
    df = classification_report_as_df(y_true, y_pred,decimal_places_for_display=None)

    assert df.shape == (6, 4)

    df_as_numpy = df.to_numpy()
    df_as_numpy[3,[0,1]]=0          # change empty strings to zeros for comparison
    expected_array = np.array([
        [0.5, 0.75, 0.6, 4],
        [0.3333333333333333, 0.5, 0.4, 2],
        [1.0, 0.5714285714285714, 0.7272727272727273, 7],
        [0, 0, 0.6153846153846154, 13],
        [0.6153846153846154, 0.6153846153846154, 0.6153846153846154, 13],
        [0.611111111111111, 0.6071428571428571, 0.5757575757575758, 13]
       ]
    )
    assert np.allclose(df_as_numpy, expected_array, atol=DIFFERENCE_TOLERANCE)


def test_styling():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 2])
    df = classification_report_as_df(y_true, y_pred,decimal_places_for_display=2)
    html = df.to_html()

    soup = BeautifulSoup(html, "html.parser")

    values = []
    for td in soup.select("td.data:not(.col3)"):
        text = td.get_text(strip=True)
        if text != "":
            values.append(text)

    pattern = re.compile(r'^\d\.\d{2}$')
    assert all(pattern.match(v) for v in values)

def test_zero_denominator_in_recall():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])

    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    assert df.loc['0', 'recall'] == 0
    assert df.loc['1', 'recall'] == 0


def test_zero_denominator_in_precision():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])

    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    assert df.loc['0', 'precision'] == 0
    assert df.loc['1', 'precision'] == 0

def test_max_recall():
    y_true = np.array([0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1])

    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    assert df.loc['1', 'recall'] == 1.0

def test_max_precision():
    y_true = np.array([0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 1, 1])

    df = classification_report_as_df(y_true, y_pred, decimal_places_for_display=None)

    assert df.loc['0', 'precision'] == 1.0
    