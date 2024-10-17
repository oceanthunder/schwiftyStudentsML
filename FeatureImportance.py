import pandas as pd

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

from sklearn.tree import DecisionTreeRegressor

dt_model = DecisionTreeRegressor(random_state=1)
dt_model.fit(X, y)

importances = dt_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeatures sorted by importance (highest to lowest):")
print(feature_importance_df)
