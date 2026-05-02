import pandas as pd
import numpy as np

def generate_data(size=200):
    np.random.seed(42)

    data = pd.DataFrame({
        'study_hours': np.random.randint(1, 10, size),
        'attendance': np.random.randint(50, 100, size),
        'sleep_hours': np.random.randint(4, 9, size),
        'previous_marks': np.random.randint(40, 100, size)
    })

    data['final_marks'] = (
        data['study_hours'] * 5 +
        data['attendance'] * 0.3 +
        data['sleep_hours'] * 2 +
        data['previous_marks'] * 0.5 +
        np.random.normal(0, 5, size)
    )

    return data
