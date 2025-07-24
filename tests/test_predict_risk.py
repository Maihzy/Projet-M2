
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.tools import predict_risk

result = predict_risk({
    "annual_inc": 40000,
    "dti": 0.33,
    "loan_amnt": 120000,
    "sub_grade_encoded": 2,
    "int_rate": 0.05,
    "term_encoded": 1,
    "mort_acc": 2,
    "revol_util": 0.8,
    "revol_bal": 10000,
    "open_acc": 5,
    "employment_length_category_Unknown": 0,
    "credit_history_months": 24,
    "total_acc": 10,
    "installment": 500,
    "home_ownership_RENT": 1
})
print("results :", result)
