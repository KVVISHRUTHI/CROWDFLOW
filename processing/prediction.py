# Store past counts
history = []

def predict_crowd(current_count):
    global history

    # add current count
    history.append(current_count)

    # keep only last 20 frames
    if len(history) > 20:
        history.pop(0)

    # if not enough data → no prediction
    if len(history) < 5:
        return current_count

    # calculate growth trend
    growth = history[-1] - history[0]

    # predicted future crowd
    predicted = current_count + growth

    # avoid negative or unrealistic drop
    return max(predicted, current_count)