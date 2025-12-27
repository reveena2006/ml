import re


class Student:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.activities = []

    def add_activity(self, activity, date, time):
        self.activities.append((activity, date, time))

    def activity_summary(self):
        logins = sum(1 for a, _, _ in self.activities if a == "LOGIN")
        submissions = sum(1 for a, _, _ in self.activities if a == "SUBMIT_ASSIGNMENT")
        return logins, submissions

STUDENT_ID_PATTERN = re.compile(r"^S\d+$")
ACTIVITY_PATTERN = re.compile(r"^(LOGIN|LOGOUT|SUBMIT_ASSIGNMENT)$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")


def read_log_file(filename):
    with open(filename, "r") as file:
        for line in file:
            try:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) != 5:
                    raise ValueError("Invalid format")

                student_id, name, activity, date, time = parts

                if not STUDENT_ID_PATTERN.match(student_id):
                    raise ValueError("Invalid Student ID")

                if not ACTIVITY_PATTERN.match(activity):
                    raise ValueError("Invalid Activity Type")

                if not DATE_PATTERN.match(date) or not TIME_PATTERN.match(time):
                    raise ValueError("Invalid Date or Time")

                yield student_id, name, activity, date, time

            except Exception as e:
                print(f"Invalid log skipped: {line.strip()} ({e})")



def main():
    students = {}
    login_tracker = {}
    daily_stats = {}

    for sid, name, activity, date, time in read_log_file("student_log.txt"):

        if sid not in students:
            students[sid] = Student(sid, name)
            login_tracker[sid] = 0

        students[sid].add_activity(activity, date, time)

        
        if activity == "LOGIN":
            login_tracker[sid] += 1
        elif activity == "LOGOUT":
            login_tracker[sid] = max(0, login_tracker[sid] - 1)

 
        daily_stats[date] = daily_stats.get(date, 0) + 1

    report = []
    report.append("STUDENT ACTIVITY REPORT\n")

    for sid, student in students.items():
        logins, submissions = student.activity_summary()
        report.append(
            f"{sid} | {student.name} | Logins: {logins} | Submissions: {submissions}"
        )

    report.append("\nABNORMAL BEHAVIOR REPORT")
    for sid, count in login_tracker.items():
        if count > 0:
            report.append(f"{sid} has multiple logins without logout")

    report.append("\nDAILY ACTIVITY STATISTICS")
    for date, count in daily_stats.items():
        report.append(f"{date}: {count} activities")

    with open("activity_report.txt", "w") as f:
        for line in report:
            f.write(line + "\n")

   
    for line in report:
        print(line)



if __name__ == "__main__":
    main()
