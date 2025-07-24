import pandas as pd
from src.models.predict import predict_batch, load_pipeline

def test_predict_batch_shapes(tmp_path):
    # tiny fake df with same schema you expect (adapt columns)
    df = pd.DataFrame({"annual_inc":[50000,60000], "loan_amnt":[10000,12000]})
    try:
        pipe = load_pipeline()
    except FileNotFoundError:
        # skip if model not trained yet
        return
    out = predict_batch(df, pipe)
    assert {"proba","label"} <= set(out.columns)
    assert len(out) == len(df)
