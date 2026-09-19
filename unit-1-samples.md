# Foundations of Machine Learning & Data Understanding

Section A: Short Answer Questions (2-3 Marks)

Q1. Define Machine Learning. How is it different from traditional programming?

**Answer:** Machine Learning (ML) is a branch of Artificial Intelligence in which computers learn patterns from data to make predictions or decisions without being explicitly programmed for every rule.

In traditional programming, a developer provides rules and input data to produce an output. In ML, the algorithm uses input data and known outputs to learn the rules (a model), which can then predict outputs for new data.

Q2. What is a learning model in Machine Learning? Explain its purpose.

**Answer:** A learning model is a mathematical representation created by an ML algorithm after studying training data. It captures the relationship or pattern between input features and the target output.

Its purpose is to generalize from past data and make predictions or decisions for unseen data. For example, a trained model can predict whether a new email is spam or not spam.

Q3. Differentiate between Supervised Learning and Unsupervised Learning with one example each.

**Answer:**

| Supervised Learning | Unsupervised Learning |
| --- | --- |
| Uses labeled data, where the correct output is known. | Uses unlabeled data, where no target output is provided. |
| Learns to predict a label or value. | Learns hidden patterns or groups in data. |
| Examples: predicting house prices; classifying emails as spam or not spam. | Examples: grouping customers by purchasing behavior; grouping similar news articles. |

Q4. What is Exploratory Data Analysis (EDA)? Why is it important before building a Machine Learning model?

**Answer:** Exploratory Data Analysis (EDA) is the process of examining, summarizing, and visualizing a dataset before model building. It uses statistics and charts such as histograms, box plots, and scatter plots to understand the data.

EDA is important because it identifies missing values, duplicate records, outliers, data distributions, and relationships between variables. These insights help in cleaning the data, selecting useful features, and choosing an appropriate model.

**Four EDA techniques are:**

1. **Summary statistics:** Measures such as mean, median, minimum, maximum, and standard deviation summarize numerical data.
2. **Missing-value analysis:** Counting null values identifies columns that require cleaning or imputation.
3. **Distribution analysis:** Histograms and bar charts show the spread, frequency, skewness, and class balance of variables.
4. **Outlier and relationship analysis:** Box plots help find unusual values, while scatter plots and correlation matrices reveal relationships between numerical variables.

Section B: Medium Answer Questions (3-4 Marks)

Q5. Explain the steps involved in the Machine Learning lifecycle/process with a suitable diagram.

**Answer:** The main steps in the Machine Learning lifecycle are:

1. **Problem definition:** Clearly define the business problem and expected output.
2. **Data collection:** Gather relevant data from databases, files, surveys, sensors, or APIs.
3. **Data preparation:** Clean missing values, remove duplicates, handle outliers, and transform data into a usable form.
4. **EDA and feature engineering:** Analyze patterns and select or create useful input features.
5. **Model selection and training:** Select a suitable algorithm and train it using training data.
6. **Model evaluation:** Measure performance using test data and metrics such as accuracy, precision, recall, MAE, or RMSE.
7. **Deployment and monitoring:** Use the model in a real application and monitor its performance over time.

```text
Problem Definition -> Data Collection -> Data Preparation -> EDA/Features
	-> Model Training -> Evaluation -> Deployment -> Monitoring
```

Q6. Compare Regression and Classification algorithms. Give one real-world application of each.

**Answer:**

| Regression | Classification |
| --- | --- |
| Predicts a continuous numerical value. | Predicts a category or class label. |
| Output can be any value within a range. | Output belongs to a fixed set of classes. |
| Example: Linear Regression. | Example: Logistic Regression or Decision Tree Classifier. |
| Application: predicting a house price. | Application: detecting whether an email is spam or not spam. |

Q7. Discuss any four common terminologies used in Machine Learning such as feature, label, dataset, and training data.

**Answer:**

1. **Feature:** An input variable used by a model for prediction. For house-price prediction, area, number of rooms, and location are features.
2. **Label (target):** The output variable that the model must predict. In the house-price example, the price is the label.
3. **Dataset:** An organized collection of data, usually arranged in rows and columns. Each row represents one observation and each column represents a variable.
4. **Training data:** The portion of a dataset used to teach the model patterns between features and labels.
5. **Test data:** A separate portion of the dataset used after training to evaluate how well the model performs on unseen data.

Section C: Long Answer Questions (4-5 Marks)

Q8. Explain the different types of Machine Learning. Discuss Supervised and Unsupervised Learning in detail with suitable examples.

**Answer:** The major types of Machine Learning are Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, and Reinforcement Learning.

**Supervised Learning** uses labeled data. Each training example contains input features and the correct output. The model learns a mapping from inputs to outputs. It is mainly used for:

- **Regression:** predicts a numerical value, such as a student's marks or house price.
- **Classification:** predicts a category, such as spam/not spam or disease/no disease.

For example, a model can be trained using previous house records containing area, location, and price. It can then predict the price of a new house.

**Unsupervised Learning** uses unlabeled data. The algorithm tries to find natural patterns, similarities, or groups without being told the correct answer. It is commonly used for:

- **Clustering:** grouping similar data points, such as grouping customers by buying habits.
- **Association:** finding items that occur together, such as products frequently purchased together.

For example, a store can use clustering to divide customers into groups such as frequent buyers, occasional buyers, and high-value buyers. These groups can be used for targeted marketing.

**Semi-Supervised Learning** uses a small amount of labeled data with a large amount of unlabeled data. **Reinforcement Learning** trains an agent through rewards and penalties, for example in game playing or robot navigation.

Q9. Describe the process of Exploratory Data Analysis (EDA). What insights can be obtained through EDA before model training?

**Answer:** EDA is a systematic process for understanding a dataset before model training. A typical EDA process includes:

1. **Inspecting the dataset:** Check the number of rows, columns, data types, and basic summary statistics.
2. **Checking data quality:** Find missing values, duplicate records, invalid values, and inconsistent formats.
3. **Studying distributions:** Use histograms and summary statistics to see whether numerical variables are normally distributed, skewed, or concentrated in a range.
4. **Detecting outliers:** Use box plots or statistical methods to find unusually high or low values.
5. **Analyzing relationships:** Use scatter plots, correlation matrices, and grouped summaries to identify relationships between features and the target.
6. **Visualizing categorical data:** Use bar charts and frequency tables to understand category counts and class imbalance.

EDA can reveal useful insights such as which features are important, whether the target classes are imbalanced, whether variables need scaling or transformation, and whether data cleaning is required. It also helps detect leakage or biased data before a model is trained.

### Programming Questions: EDA Process (4 Marks Each)

Q10. Write a Python program using pandas to perform initial EDA on a CSV file named `students.csv`. Display the first five rows, dataset information, summary statistics, and the number of missing values in each column.

**Solution:**

```python
import pandas as pd

data = pd.read_csv("students.csv")

print("First five rows:")
print(data.head())

print("\nDataset information:")
data.info()

print("\nSummary statistics:")
print(data.describe())

print("\nMissing values in each column:")
print(data.isnull().sum())
```

This program helps identify the dataset structure, data types, numerical summaries, and missing values before model training.

Q11. Write a Python program to visualize the distribution of a numerical column named `marks` and detect its outliers using a box plot.

**Solution:**

```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("students.csv")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(data["marks"].dropna(), bins=10, edgecolor="black")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.boxplot(data["marks"].dropna())
plt.title("Outliers in Marks")
plt.ylabel("Marks")

plt.tight_layout()
plt.show()
```

The histogram shows the distribution of marks, while values plotted beyond the whiskers of the box plot may be treated as outliers for further investigation.

Q12. Consider a dataset `students.csv` containing student information and placement status. Write a Python program using pandas to load the dataset, display the first five records, and separate the input features from the target variable `Placement_Status`.

**Solution:**

```python
import pandas as pd

data = pd.read_csv("students.csv")

print("First five records:")
print(data.head())

features = data.drop("Placement_Status", axis=1)
target = data["Placement_Status"]

print("\nInput features:")
print(features.head())

print("\nTarget variable:")
print(target.head())
```

Here, `features` contains all columns except `Placement_Status`, and `target` contains the placement-status values to be predicted.

Q13. Discuss any five real-world applications of Machine Learning. Explain how Machine Learning helps solve problems in these domains.

**Answer:**

1. **Healthcare:** ML helps analyze medical images, predict disease risk, and support early diagnosis. For example, it can identify patterns in X-rays that may indicate pneumonia.
2. **Finance:** Banks use ML for fraud detection, credit scoring, and risk assessment. A model can flag unusual transactions in real time.
3. **E-commerce:** Online stores use recommendation systems to suggest products based on browsing and purchase history. This improves customer experience and sales.
4. **Transportation:** ML is used in route optimization, traffic prediction, and self-driving vehicle systems. It helps reduce travel time and improve safety.
5. **Education:** ML can personalize learning by recommending lessons based on a student's performance and identifying students who may need extra support.

In each domain, ML learns from historical data to automate decisions, identify patterns quickly, and provide predictions that assist people and organizations.

### MCQ-Based Questions (Optional)

Which type of learning uses labeled data?

a) Reinforcement Learning
b) Supervised Learning
c) Unsupervised Learning
d) Semi-Supervised Learning

Predicting house prices is an example of:

a) Classification
b) Clustering
c) Regression
d) Association

Which technique is commonly used in EDA to understand data distribution?

a) Histogram
b) Encryption
c) Compilation
d) Hashing

Grouping customers based on purchasing behavior is an example of:

a) Regression
b) Classification
c) Clustering
d) Optimization

In Machine Learning, a variable used as input to a model is called:

a) Label
b) Target
c) Feature
d) Prediction

**Answer Key:**

1. b) Supervised Learning
2. c) Regression
3. a) Histogram
4. c) Clustering
5. c) Feature