def analyze_risk(count, future_count, congestion):

    if congestion > 7:
        return "CRITICAL"

    elif future_count > count + 20:
        return "SURGE"

    elif count > 80:
        return "HIGH"

    else:
        return "SAFE"