# classification-report-as-df
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://github.com/Michal58/classification-report-as-df/actions/workflows/tests.yml/badge.svg)

Small, single-module utility that converts scikit-learn's text-style classification report into an easy-to-manipulate pandas `DataFrame`.
For easier display in IPython environments (e.g. Jupyter), the report can also be returned as a pandas `Styler`.

## Example usage
General scenario:
```python
import numpy as np
from classification_report_as_df import classification_report_as_df

y_true = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0, 0, 0, 1, 1])

df = classification_report_as_df(
    y_true=y_true,
    y_pred=y_pred
)
```
Jupyter notebook pretty-rendering scenario:
```python
import numpy as np
from classification_report_as_df import classification_report_as_df

y_true = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0, 0, 0, 1, 1])

classification_report_as_df(
    y_true=y_true,
    y_pred=y_pred,
    decimal_places_for_display=2
)
```

## Development
Clone repository:
```
git clone https://github.com/Michal58/classification-report-as-df
cd classification-report-as-df
```
Install dependencies:
```
pip install classification_report_as_df[dev]
```
or (from root directory):
```
pip install -e '.[dev]'
```

You can run tests calling (from root directory): 
```
pytest
```
## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
