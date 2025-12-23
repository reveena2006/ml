
student_ids = ('S101', 'S102', 'S103', 'S104')


students = {
    'S101': {'name': 'Asha', 'assignment': 78, 'test': 80, 'attendance': 92, 'hours': 8},
    'S102': {'name': 'Ravi', 'assignment': 65, 'test': 68, 'attendance': 85, 'hours': 5},
    'S103': {'name': 'Meena', 'assignment': 88, 'test': 90, 'attendance': 96, 'hours': 10},
    'S104': {'name': 'Kiran', 'assignment': 55, 'test': 58, 'attendance': 78, 'hours': 4}
}


def calculate_average(assignment, test):
    return (assignment + test) / 2


def determine_risk(avg, attendance, hours):
    if avg >= 75 and attendance >= 90 and hours >= 8:
        return "Low Risk"
    elif avg >= 60 and attendance >= 80:
        return "Moderate Risk"
    else:
        return "High Risk"


print("STUDENT PERFORMANCE REPORT")
print("-" * 50)


for sid in student_ids:
    data = students[sid]

    avg_score = calculate_average(data['assignment'], data['test'])
    risk_level = determine_risk(avg_score, data['attendance'], data['hours'])

    print("Student ID     :", sid)
    print("Name           :", data['name'])
    print("Assignment     :", data['assignment'])
    print("Test Score     :", data['test'])
    print("Average Score  :", round(avg_score, 2))
    print("Attendance     :", data['attendance'], "%")
    print("Study Hours    :", data['hours'], "hrs/week")
    print("Risk Level     :", risk_level)
    if risk_level == "High Risk":
        print("Academic Support : REQUIRED")
    print("-" * 50)
