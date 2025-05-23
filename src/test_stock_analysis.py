import pytest
import pandas as pd
import numpy as np
from src.Stock_Analysis import HMMStockPredictor

def test_extract_features():
    data = pd.DataFrame({
        'Open': [100, 102],
        'Close': [101, 103],
        'High': [102, 104],
        'Low': [99, 101]
    })
    features = HMMStockPredictor._extract_features(data)
    assert features.shape == (2, 3)
    np.testing.assert_almost_equal(features[0, 0], 0.01, decimal=2)

def test_model_fit_and_predict():
    # Minimal mock data
    data = pd.DataFrame({
        'Open': np.linspace(100, 110, 20),
        'Close': np.linspace(101, 111, 20),
        'High': np.linspace(102, 112, 20),
        'Low': np.linspace(99, 109, 20),
        'Volume': np.ones(20),
        'Adj Close': np.linspace(101, 111, 20)
    }, index=pd.date_range('2022-01-01', periods=20))
    data.to_csv('mock_data.csv')
    # Patch data.DataReader to return our mock data
    import types
    def mock_datareader(*args, **kwargs):
        return data
    import src.Stock_Analysis as sa
    sa.data.DataReader = mock_datareader
    predictor = sa.HMMStockPredictor('MOCK', '2022-01-01', '2022-01-20', future_days=2, train_size=0.5)
    predictor.fit()
    preds = predictor.predict_close_prices_for_period()
    assert len(preds) == predictor.days
    assert all(isinstance(x, float) for x in preds) 