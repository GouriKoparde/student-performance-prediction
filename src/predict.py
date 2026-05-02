import pandas as pd
import joblib

def load_model():
    model = joblib.load("models/student_model.pkl")
    return model

def predict_student(model):
    new_student = pd.DataFrame({
        'study_hours': [6],
        'attendance': [85],
        'sleep_hours': [7],
        'previous_marks': [75]
    })

    prediction = model.predict(new_student)

    return prediction[0]