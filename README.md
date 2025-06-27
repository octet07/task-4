# Breast Cancer Classification using Logistic Regression

This project demonstrates a full binary classification workflow using the Breast Cancer Wisconsin Dataset. The objective is to classify tumors as either malignant (0) or benign (1) based on various medical features.

---

## Workflow Summary

### 1. Load the Dataset

We use `sklearn.datasets.load_breast_cancer()` to load a built-in dataset with 569 samples and 30 features. The target variable is binary:

- 0 → Malignant  
- 1 → Benign

### 2. Train-Test Split

The dataset is split into training and testing sets using an 80-20 split to evaluate model generalization.

### 3. Standardize Features

We use `StandardScaler` to normalize the data so that features have zero mean and unit variance. This is essential for models like Logistic Regression that are sensitive to feature scales.

### 4. Train a Logistic Regression Model

We use `LogisticRegression` from `sklearn.linear_model` to fit a binary classification model on the training data.

### 5. Evaluate the Model

We evaluate the model using:

- Confusion Matrix to view true positives, false positives, true negatives, and false negatives  
- Precision, Recall, and F1-score to understand classification performance  
- ROC Curve and AUC Score to visualize the trade-off between sensitivity and specificity

### 6. Tune the Classification Threshold

By default, Logistic Regression predicts class 1 if the probability is greater than 0.5. We manually tune the threshold to 0.3 to increase recall and reduce false negatives. This is important in medical diagnosis tasks where catching all positive cases (malignant tumors) is critical.

## Conclusion

This project demonstrates how to:

- Prepare and scale data for Logistic Regression  
- Evaluate classification performance with multiple metrics  
- Adjust classification thresholds for real-world use cases  

This setup is flexible and can be extended to other models like SVM, Decision Trees, or ensemble methods.
