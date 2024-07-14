from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Upload the file
uploaded = files.upload()

# Assuming the file name is 'employee_burnout_analysis-AI.xlsx'
file_name = 'employee_burnout_analysis-AI.xlsx'
xls = pd.ExcelFile(file_name)

# Automatically select the first sheet
sheet_name = xls.sheet_names[0]

# Read the chosen sheet
data = pd.read_excel(file_name, sheet_name=sheet_name)

# Handle missing values by filling them with the mean of the column
data['Resource Allocation'].fillna(data['Resource Allocation'].mean(), inplace=True)
data['Mental Fatigue Score'].fillna(data['Mental Fatigue Score'].mean(), inplace=True)

# Convert Burn Rate to a binary target
data['Burnout'] = np.where(data['Burn Rate'] > 0.5, 1, 0)

# Drop irrelevant columns
data.drop(columns=['Employee ID', 'Date of Joining', 'Burn Rate'], inplace=True)

# Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Split the data into features and target
X = data_encoded.drop('Burnout', axis=1)
y = data_encoded['Burnout']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Identify top burnout employees (assuming 'Mental Fatigue Score' is a key indicator)
top_burnout_employees = data[data['Burnout'] == 1].sort_values(by='Mental Fatigue Score', ascending=False).head(10)

# Display the analysis table
print("Top 10 Burnout Employees based on Mental Fatigue Score:")
print(top_burnout_employees)

# Generate a bar graph for Mental Fatigue Score vs Resource Allocation
plt.figure(figsize=(10, 6))
bar_width = 0.35
indices = np.arange(top_burnout_employees.shape[0])
plt.bar(indices, top_burnout_employees['Mental Fatigue Score'], bar_width, color='red', alpha=0.7, label='Mental Fatigue Score')
plt.bar(indices + bar_width, top_burnout_employees['Resource Allocation'], bar_width, color='blue', alpha=0.5, label='Resource Allocation')
plt.xlabel('Employee Index')
plt.ylabel('Scores')
plt.title('Top Burnout Employees Analysis')
plt.xticks(indices + bar_width / 2, top_burnout_employees.index, rotation=45)
plt.legend()
plt.show()

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No Burnout', 'Burnout'])
plt.show()
