from src.data_generation import generate_data
from src.model import train_model
from src.predict import predict_student, load_model

import os

# ---------------------------
# CHECK IF MODEL EXISTS
# ---------------------------
if not os.path.exists("models/student_model.pkl"):
    print("Training model...")

    data = generate_data()
    model, rmse = train_model(data)

    print("Model trained. RMSE:", rmse)

else:
    print("Loading saved model...")
    model = load_model()

# ---------------------------
# MAKE PREDICTION
# ---------------------------
result = predict_student(model)

print("\nPredicted Marks:", result)

# ---------------------------
# SAVE OUTPUT
# ---------------------------
import pandas as pd

output = pd.DataFrame({
    "Predicted Marks": [result]
})

output.to_csv("outputs/prediction.csv", index=False)

print("Prediction saved to outputs/prediction.csv")
# ---------------------------
# SAVE VISUALIZATION
# ---------------------------
import matplotlib.pyplot as plt

plt.figure()
plt.bar(["Predicted Marks"], [result])
plt.title("Student Performance Prediction")

plt.savefig("outputs/prediction_chart.png")

print("Chart saved to outputs/prediction_chart.png")