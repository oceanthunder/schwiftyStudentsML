import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Input

path = 'StudentPerformanceFactors.csv'
student_performance = pd.read_csv(path)
student_performance.head()


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


scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)


train_X = train_X.reshape(train_X.shape[0], 40, 1, 1)
val_X = val_X.reshape(val_X.shape[0], 40, 1, 1)


model = models.Sequential([
    Input(shape=(40, 1, 1)),
    layers.Conv2D(32, (2, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Conv2D(64, (2, 1), activation='relu'),
    layers.MaxPooling2D((2, 1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


history = model.fit(
    train_X, 
    train_y, 
    epochs=100, 
    validation_data=(val_X, val_y) 
)


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epochs, mae, label='Training MAE')
plt.plot(epochs, val_mae, label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

val_loss, val_mae = model.evaluate(val_X, val_y)
print(f"Validation Loss: {val_loss}, Validation MAE: {val_mae}")
