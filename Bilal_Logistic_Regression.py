import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt

# Define the dataset
data = {
    'Feature1': [1, 2, 3, 4, 6, 6, 7, 8, 9, 10],
    'Feature2': [1, 4, 5, 6, 6, 8, 9, 10, 11, 12],
    'Target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Split the dataset into features (X) and target variable (y)
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
recall = recall_score(y, y_pred)
precision = precision_score(y, y_pred)
f1 = f1_score(y, y_pred)
confusion_mat = confusion_matrix(y, y_pred)

# Print the evaluation metrics
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)
print('Confusion Matrix:')
print(confusion_mat)

# Plot the confusion matrix
labels = ['Negative', 'Positive']
fig, ax = plt.subplots()
plt.imshow(confusion_mat, cmap=plt.cm.Blues)
plt.colorbar()

# Add labels, title, and ticks
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values')
ax.set_title('Confusion Matrix')
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Loop over data dimensions and create text annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, confusion_mat[i, j], ha='center', va='center', color='white')

plt.show()

# Plot the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, y_pred)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
