import joblib

def load_model():
    model_path = 'C:/Users/HP/Desktop/EstateVerse-Base-Model/backend/app/models/xgboost_model.pkl'
    return joblib.load(model_path)
