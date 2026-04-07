def calculate_rule_score(data):

    score = 0
    reasons = []

    # Study Hours
    if data['study_hours'] >= 10:
        score += 20
    elif data['study_hours'] >= 7:
        score += 15
    elif data['study_hours'] >= 4:
        score += 10
    elif data['study_hours'] >= 2:
        score += 5
    else:
        score -= 15
        reasons.append("Very low study hours")

    # Attendance
    if data['attendance'] >= 90:
        score += 20
    elif data['attendance'] >= 75:
        score += 15
    elif data['attendance'] >= 50:
        score += 8
    elif data['attendance'] >= 30:
        score += 3
    else:
        score -= 20
        reasons.append("Poor attendance")

    # Previous Score
    if data['previous_score'] >= 90:
        score += 20
    elif data['previous_score'] >= 75:
        score += 15
    elif data['previous_score'] >= 60:
        score += 10
    elif data['previous_score'] >= 40:
        score += 5
    else:
        score -= 15
        reasons.append("Weak academic history")

    score += data['assignments'] * 0.05
    score += data['internal_marks'] * 0.05

    if 6 <= data['sleep_hours'] <= 8:
        score += 5
    else:
        score -= 5

    if data['internet_usage'] > 8:
        score -= 10
        reasons.append("Too much internet usage")

    if data['extra_activities'] == "Yes":
        score += 2
    # 🔥 HIGH PERFORMANCE BOOST
    if (
    data['study_hours'] >= 8 and
    data['attendance'] >= 85 and
    data['previous_score'] >= 80
    ):
        score += 20
    return max(0, min(100, score)), reasons