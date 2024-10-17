import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


path = 'StudentPerformanceFactors.csv'
student_performance = pd.read_csv(path)


print(student_performance.isnull().sum())
print(student_performance.describe())


y = student_performance['Exam_Score']
features = [
    'Hours_Studied', 'Previous_Scores', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions',
    'Physical_Activity', 'Parental_Involvement', 'Gender', 'Access_to_Resources',
    'Extracurricular_Activities', 'Motivation_Level', 'Internet_Access', 'Family_Income',
    'Teacher_Quality', 'School_Type', 'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home'
]
X = pd.get_dummies(student_performance[features], drop_first=True)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
val_X = scaler.transform(val_X)


train_X = train_X.reshape(train_X.shape[0], -1)
val_X = val_X.reshape(val_X.shape[0], -1)


model = models.Sequential([
    Input(shape=(train_X.shape[1],)),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(1)  # Output layer
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics=['mae'])


early_stopping = EarlyStopping(
    monitor='val_mae',
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)


history = model.fit(
    train_X,
    train_y,
    epochs=200,
    batch_size=32,
    validation_data=(val_X, val_y),
    callbacks=[early_stopping]
)


val_predictions = model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE: {val_mae}")


import matplotlib.pyplot as plt

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid()
plt.show()
