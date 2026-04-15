def check_alert(current, predicted):
    growth = predicted - current

    if predicted > 150 or growth > 40:
        return "HIGH RISK"
    elif predicted > 70:
        return "MODERATE"
    else:
        return "SAFE"
