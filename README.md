# Loan Prediction - Classification using PySpark ML

## Project Information

The goal of this project is to demonstrate the use of PySpark and Machine Learning to predict loan approvals. The project involves the following steps:

1. **Data Collection**: Gathering data related to loan approvals, including features such as income, credit score, loan amount, and loan status.
2. **Data Preparation**: Preprocessing and cleaning the data to ensure it is ready for use in Machine Learning algorithms.
3. **Feature Engineering**: Selecting and engineering relevant features to improve the model's accuracy.
4. **Model Training**: Using PySpark to train a Machine Learning model on the prepared data.
5. **Model Evaluation**: Testing the trained model on a validation set to evaluate its accuracy and adjust parameters if necessary.
6. **Loan Prediction**: Using the trained model to predict whether a loan will be approved based on input features.

The end result is a Machine Learning model that can accurately predict loan approvals based on various features. This can be used by financial institutions or individuals to make more informed decisions regarding loans. Additionally, the project can be extended by exploring different Machine Learning algorithms and improving the model's accuracy.

## Dataset Information

Dream Housing Finance company deals in all home loans. They have a presence across all urban, semi-urban, and rural areas. Customers first apply for a home loan, after which the company validates the customer's eligibility. The company wants to automate the loan eligibility process (in real-time) based on customer details provided while filling out the online application form. These details include Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. To automate this process, they have given a problem to identify customer segments eligible for loan amounts so that they can specifically target these customers.

This is a standard supervised classification task where we have to predict whether a loan would be approved or not. Below are the dataset attributes with descriptions:

| Variable          | Description                              |
|-------------------|------------------------------------------|
| Loan_ID           | Unique Loan ID                           |
| Gender            | Male/Female                              |
| Married           | Applicant married (Y/N)                  |
| Dependents        | Number of dependents                     |
| Education         | Applicant Education (Graduate/Undergraduate) |
| Self_Employed     | Self-employed (Y/N)                      |
| ApplicantIncome   | Applicant income                         |
| CoapplicantIncome | Coapplicant income                       |
| LoanAmount        | Loan amount in thousands                 |
| Loan_Amount_Term  | Term of loan in months                   |
| Credit_History    | Credit history meets guidelines          |
| Property_Area     | Urban/Semi-Urban/Rural                   |
| Loan_Status       | Loan approved (Y/N)                      |

**Download link:** [Kaggle Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)

## Libraries

- PySpark

## Algorithms

- Logistic Regression
- Random Forest

**Best Model AUC:** 83.00

## Results

model achieved an accuracy of **83%** on the test data. While this is a strong result, there is always room for improvement. Future work will focus on exploring additional feature engineering techniques, trying different models, and performing more extensive hyperparameter tuning.

## Conclusion

This project demonstrates the application of PySpark ML for building a classification model to predict loan approval status. The process includes data preprocessing, feature engineering, model training, and evaluation, showcasing the capabilities of PySpark in handling large datasets and complex machine learning tasks.

## Future Work

- Experiment with other machine learning algorithms such as Gradient Boosting and XGBoost.
- Perform more extensive hyperparameter tuning.
- Explore additional feature engineering techniques to improve model performance.