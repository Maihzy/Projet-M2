from modules.tools import predict_risk

def test_model_availability():
    test_data = {
        'sub_grade_encoded': 10,
        'dti': 25.0,
        'annual_inc': 50000,
        'int_rate': 12.0,
        'term_encoded': 36,
        'loan_amnt': 20000,
        'mort_acc': 2,
        'revol_util': 50.0,
        'revol_bal': 15000,
        'open_acc': 8,
        'employment_length_category_Unknown': 0,
        'credit_history_months': 120,
        'total_acc': 20,
        'installment': 400,
        'home_ownership_RENT': 1
    }
    proba, _ = predict_risk(test_data)
    assert isinstance(proba, float)