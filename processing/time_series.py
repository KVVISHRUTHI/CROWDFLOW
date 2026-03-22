import numpy as np

def predict_future_crowd(count_history):

    if len(count_history) < 10:
        return count_history[-1] if count_history else 0

    x = np.arange(len(count_history))
    y = np.array(count_history)

    # 🔥 Linear regression
    coeffs = np.polyfit(x, y, 1)

    slope = coeffs[0]
    intercept = coeffs[1]

    future_x = len(count_history) + 5

    future = int(slope * future_x + intercept)

    return max(future, 0)