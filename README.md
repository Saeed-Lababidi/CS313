# CS313
This is the repo for our Introduction to Data Science Course
# Online Shoppers Purchasing Intention Prediction

This project aims to predict whether an online shopper will generate revenue for the company. The dataset contains various attributes of online shopping sessions, and the target variable is `Revenue`, indicating whether a session generated revenue or not.

## Dataset

The dataset used in this project is the "Online Shoppers Purchasing Intention Dataset" available from the UCI Machine Learning Repository. The dataset contains 12,330 instances and 18 features.

## Project Structure

- **Cell 1:** Mount Google Drive to access the dataset.
- **Cell 2:** Load the dataset using pandas.
- **Cell 3:** Check for and handle missing values.
- **Cell 4:** Handle outliers in numerical columns.
- **Cell 5:** Create a new feature `Total_Duration` which is the sum of `Administrative_Duration`, `Informational_Duration`, and `ProductRelated_Duration`.
- **Cell 6:** Encode categorical variables and normalize numerical variables.
- **Cell 7:** Perform exploratory data analysis (EDA) including a correlation heatmap and distribution plot of the target variable.
- **Cell 8:** Split the data into training and testing sets.
- **Cell 9:** Train a Random Forest Classifier and evaluate its performance.
- **Cell 10:** Optimize the Random Forest Classifier using Grid Search and evaluate the optimized model.

## How to Run

1. **Clone the Repository:**
   git clone https://github.com/yourusername/online-shoppers-intention-prediction.git
   cd online-shoppers-intention-prediction
   
**2. Open the Google Colab Notebook:**
Open Google Colab.
Upload the provided notebook file or copy the code into a new Colab notebook.

**3. Mount Google Drive:**
Ensure your dataset is in your Google Drive.
Run the cells to mount Google Drive and load the dataset.

**4. Run the Notebook Cells:**
Execute each cell sequentially to preprocess the data, perform EDA, train the model, and evaluate the results.

**Dependencies**
Python 3.7+
pandas
numpy
seaborn
matplotlib
scikit-learn
Ensure all dependencies are installed. You can install them using pip:
pip install pandas numpy seaborn matplotlib scikit-learn
**Results**
The model performance will be displayed at the end of the notebook, including the classification report and confusion matrix for both the initial and optimized models.
