from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def impute_and_scale(X: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray, StandardScaler, IterativeImputer]:
    # Ensure numeric
    X_num = X.apply(pd.to_numeric, errors="coerce")

    # Drop columns with extreme missingness (>50%)
    missing_frac = X_num.isna().mean()
    keep_cols = missing_frac[missing_frac <= 0.5].index.tolist()
    X_num = X_num[keep_cols]

    # Impute
    imputer = IterativeImputer(random_state=random_state, sample_posterior=False, max_iter=15, initial_strategy="median")
    X_imputed = imputer.fit_transform(X_num.values)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    X_imp_df = pd.DataFrame(X_imputed, index=X.index, columns=keep_cols)
    return X_imp_df, X_scaled, scaler, imputer
