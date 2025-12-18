mkdir "$(dirname "$0")/deployment"
cd "$(dirname "$0")/deployment"
python3 -m venv venv
venv/bin/pip install ../../dist/classification_report_as_df-0.1.0-py3-none-any.whl
venv/bin/python - <<EOF
import numpy as np
from classification_report_as_df import classification_report_as_df

y_true = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0, 0, 0, 1, 1])

df = classification_report_as_df(y_true=y_true, y_pred=y_pred)

print("DataFrame output:")
print(df)
EOF

cd -
rm -r deployment