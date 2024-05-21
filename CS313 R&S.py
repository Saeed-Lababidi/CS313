

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Dataset/online_shoppers_intention.csv')
data.head()

# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values (if any)
data = data.dropna()

# Creating a Total_Duration feature
data['Total_Duration'] = data['Administrative_Duration'] + data['Informational_Duration'] + data['ProductRelated_Duration']
data.head()

from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Normalize numerical variables
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

data.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Distribution of the target variable
sns.countplot(x='Revenue', data=data)
plt.show()

from sklearn.model_selection import train_test_split

X = data.drop('Revenue', axis=1)
y = data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Balanced Random Forest Classifier
model = BalancedRandomForestClassifier(random_state=42)
model.fit(X_train_smote, y_train_smote)

y_pred = model.predict(X_test)

# Evaluate the model
print("Balanced Random Forest Classifier:")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


from sklearn.model_selection import GridSearchCV

# Grid search for Balanced Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train_smote, y_train_smote)

best_model = grid_search.best_estimator_

y_pred_optimized = best_model.predict(X_test)

# Evaluate the optimized model
print("Optimized Balanced Random Forest Classifier:")
print(classification_report(y_test, y_pred_optimized))
print(confusion_matrix(y_test, y_pred_optimized))

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the model again with the balanced dataset
model_smote = RandomForestClassifier(random_state=42)
model
