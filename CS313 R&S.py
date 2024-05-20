#This project has been implemented using google colab
#Cell 1
from google.colab import drive
drive.mount('/content/drive')
#Cell 2
import pandas as pd
# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Dataset/online_shoppers_intention.csv')
data.head()
#Cell 3
# Check for missing values
data.isnull().sum()

# Drop rows with missing values (if any)
data = data.dropna()
# Cell 4
# Handle outliers for numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
#Cell 5
# Creating a Total_Duration feature
data['Total_Duration'] = data['Administrative_Duration'] + data['Informational_Duration'] + data['ProductRelated_Duration']
data.head()
#Cell 6
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
#Cell 7
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# Distribution of the target variable
sns.countplot(x='Revenue', data=data)
plt.show()
#Cell 8
from sklearn.model_selection import train_test_split

X = data.drop('Revenue', axis=1)
y = data['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Cell 9
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#Cell 10
from sklearn.model_selection import GridSearchCV

# Grid search for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred_optimized = best_model.predict(X_test)

# Evaluate the optimized model
print(classification_report(y_test, y_pred_optimized))
print(confusion_matrix(y_test, y_pred_optimized))
