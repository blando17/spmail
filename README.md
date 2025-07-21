#  Spam Mail Detector

A machine learning project that detects spam emails based on their content using **Logistic Regression** and **TF-IDF vectorization**. The model achieves high accuracy in classifying messages as either **spam** or **ham** (not spam).

---

##  Project Overview

This project solves a text classification problem by identifying whether a message is spam or not. The dataset used contains labeled SMS messages. After preprocessing and feature extraction, we train a logistic regression model and evaluate its performance.

---

## Technologies Used

| Tool/Library         | Purpose                             |
|----------------------|-------------------------------------|
| Python               | Programming Language                |
| Pandas & NumPy       | Data loading and manipulation       |
| scikit-learn         | Preprocessing, vectorization, modeling |
| TfidfVectorizer      | Feature extraction from text        |
| LogisticRegression   | Model for binary classification     |

---

##  Dataset Description

- **Dataset Size**: 5,572 messages
- **Columns**:
  - `Category`: spam or ham (target)
  - `Message`: the text of the message

---

##  Data Preprocessing

- **Replaced null values** with empty strings.
- **Label Encoding**:
  - spam → 0
  - ham → 1

```python
mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})
```

- **Feature/Target Split**:
```python
X = mail_data['Message']
y = mail_data['Category']
```

---

##  Feature Extraction (TF-IDF)

TF-IDF is used to convert text data into numerical vectors:

```python
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(x_train)
X_test_features = feature_extraction.transform(x_test)
```

---

##  Model: Logistic Regression

```python
regressor = LogisticRegression()
regressor.fit(X_train_features, Y_train)
```

---

##  Evaluation Results

- **Training Accuracy**: `96.76%`
- **Testing Accuracy**: `96.68%`

Evaluated using `accuracy_score` from `sklearn.metrics`.

---

##  Prediction Example

```python
input_mail = ["I HAVE A DATE ON SUNDAY WITH WILL!!"]
input_data_features = feature_extraction.transform(input_mail)
prediction = regressor.predict(input_data_features)

if prediction[0] == 1:
    print("Ham mail")
else:
    print("Spam mail")
```

**Output**: `Ham mail`
