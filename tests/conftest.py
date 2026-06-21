import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "age": np.random.randint(20, 60, 100),
        "income": np.random.randn(100) * 20000 + 50000,
        "city": np.random.choice(["NYC", "LA", "Chicago"], 100),
        "target": (np.random.randn(100) > 0).astype(int),
    })
