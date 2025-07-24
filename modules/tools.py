import pandas as pd
import joblib

try:
    _artifact = joblib.load("DATA/models/final_model.pkl")
except FileNotFoundError:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
    _artifact = RandomForestClassifier(random_state=42)
    _artifact.fit(X, y)

try:
    _features = joblib.load("DATA/models/final_features.pkl")
except:
    _features = [f"feature_{i}" for i in range(15)]

if isinstance(_artifact, dict) and 'model' in _artifact:
    _model = _artifact['model']
else:
    _model = _artifact

def predict_risk(data: dict):
    try:
        df_data = {feature: data.get(feature, 0.5) for feature in _features}
        df = pd.DataFrame([df_data])
        X = df[_features]
        if hasattr(_model, 'predict_proba'):
            proba = _model.predict_proba(X)[:, 1][0]
        else:
            proba = _model.predict(X)[0]
            proba = max(0, min(1, proba))
        pred = int(proba >= 0.4)
        return float(proba), int(pred)
    except Exception:
        return 0.5, 0

def explain_risk(data: dict):
    try:
        import shap
        df_data = {feature: data.get(feature, 0.5) for feature in _features}
        df = pd.DataFrame([df_data])
        X = df[_features]
        explainer = shap.TreeExplainer(_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[-1][0]
        else:
            shap_vals = shap_vals[0]
        return list(zip(_features, shap_vals.tolist()))
    except Exception:
        return [(feature, 0.0) for feature in _features]

def predict_risk_score(client_data):
    try:
        model_data = prepare_data_for_model(client_data)
        proba, _ = predict_risk(model_data)
        return {
            'score': proba,
            'input_data': client_data.copy(),
            'model_used': 'LightGBM_REEL'
        }
    except Exception:
        return predict_risk_simulation(client_data)

def prepare_data_for_model(client_data):
    dti = client_data.get('dti', 25.0)
    annual_inc = client_data.get('annual_inc', 50000)
    loan_amnt = client_data.get('loan_amnt', 20000)
    int_rate = client_data.get('int_rate', 12.0)
    term = client_data.get('term', 36)
    revol_util = client_data.get('revol_util', 50.0)
    return {
        'dti': dti,
        'annual_inc': annual_inc,
        'loan_amnt': loan_amnt,
        'int_rate': int_rate,
        'revol_util': revol_util,
        'term_encoded': term,
        'installment': loan_amnt / max(term, 1),
        'sub_grade_encoded': estimate_sub_grade(dti, annual_inc, int_rate),
        'mort_acc': 2,
        'revol_bal': min(annual_inc * 0.3, 25000),
        'open_acc': max(5, min(15, int(annual_inc / 10000))),
        'employment_length_category_Unknown': 0,
        'credit_history_months': max(24, min(240, int(annual_inc / 1000))),
        'total_acc': max(8, min(30, int(annual_inc / 5000))),
        'home_ownership_RENT': 1 if annual_inc < 75000 else 0
    }

def estimate_sub_grade(dti, annual_inc, int_rate):
    score = 0
    if dti < 15: score += 3
    elif dti < 25: score += 2
    elif dti < 35: score += 1
    else: score -= 1
    if annual_inc > 75000: score += 3
    elif annual_inc > 50000: score += 2
    elif annual_inc > 30000: score += 1
    else: score -= 1
    if int_rate < 8: score += 2
    elif int_rate < 12: score += 1
    elif int_rate > 18: score -= 2
    elif int_rate > 15: score -= 1
    return max(1, min(20, 10 + score))

def predict_risk_simulation(client_data):
    dti = client_data.get('dti', 25.0)
    annual_inc = client_data.get('annual_inc', 50000)
    loan_amnt = client_data.get('loan_amnt', 20000)
    int_rate = client_data.get('int_rate', 12.0)
    revol_util = client_data.get('revol_util', 50.0)
    score = 0.0
    if dti < 10: score += 0.05
    elif dti < 20: score += 0.15
    elif dti < 30: score += 0.25
    elif dti < 40: score += 0.45
    else: score += 0.65
    if annual_inc > 100000: score += 0.02
    elif annual_inc > 75000: score += 0.08
    elif annual_inc > 50000: score += 0.15
    elif annual_inc > 30000: score += 0.25
    else: score += 0.35
    ratio = loan_amnt / max(annual_inc, 1)
    if ratio < 0.2: score += 0.02
    elif ratio < 0.4: score += 0.08
    elif ratio < 0.6: score += 0.15
    elif ratio < 0.8: score += 0.25
    else: score += 0.35
    if int_rate < 6: score += 0.01
    elif int_rate < 10: score += 0.03
    elif int_rate < 15: score += 0.06
    elif int_rate < 20: score += 0.10
    else: score += 0.15
    if revol_util < 30: score += 0.02
    elif revol_util < 60: score += 0.05
    else: score += 0.08
    final_score = max(0.05, min(0.95, score))
    return {
        'score': final_score,
        'input_data': client_data.copy(),
        'model_used': 'SIMULATION_FALLBACK'
    }
