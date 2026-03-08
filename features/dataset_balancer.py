from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np


def balance_dataset(X, y):
    """
    Balance dataset using SMOTE. Prints class counts for visibility.
    """
    print("Before balancing:", Counter(y))
    smote = SMOTE()
    X_bal, y_bal = smote.fit_resample(X, y)
    print("After balancing:", Counter(y_bal))
    return np.asarray(X_bal), np.asarray(y_bal)
