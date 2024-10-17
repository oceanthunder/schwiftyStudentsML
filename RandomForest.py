import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

path = 'StudentPerformanceFactors.csv'
student_performance = pd.read_csv(path)

y = student_performance.Exam_Score
features = [
    'Hours_Studied', 'Previous_Scores', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions',
    'Physical_Activity', 'Parental_Involvement', 'Gender', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type',
    'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home'
]
X = student_performance[features]
X = pd.get_dummies(X)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state = 1)
rf_model.fit(train_X, train_y)
val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
