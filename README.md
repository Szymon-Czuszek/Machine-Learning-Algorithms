# 🤖 Machine Learning Algorithms

👋 Welcome to the Machine Learning Algorithms repository! This repository is dedicated to providing comprehensive examples and implementations of various machine learning algorithms. All content in this repository is licensed under the MIT License.

## 📌 Contents

### 🔍 KNN

#### 🏦 K-Nearest Neighbors (KNN) in Detecting Fake Banknotes
K-Nearest Neighbors (KNN) is a simple yet effective machine learning algorithm used for classification and regression tasks. In the context of detecting fake banknotes, KNN helps classify new banknotes as either genuine or counterfeit based on their similarity to known examples.

##### ⚙️ How KNN Works
KNN is a non-parametric, instance-based learning algorithm, meaning it does not make strong assumptions about the data distribution and instead stores all training data to make predictions when needed. The core idea behind KNN is as follows:
1. 📏 Distance Measurement:
- When given a new banknote, KNN computes its distance from all training samples using a distance metric, typically Euclidean distance.
2. 🎯 Selecting K Neighbors:
- The algorithm selects the K nearest data points based on the calculated distances.
3. 🗳️ Voting Mechanism:
- In classification tasks (like fake vs. genuine banknotes), KNN assigns the class label based on the majority vote among the K nearest neighbors.
4. ✅ Final Classification:
- The new banknote is classified as genuine or counterfeit depending on which class is more frequent among its neighbors.

##### 🎯 Why KNN for Fake Banknote Detection?
Detecting counterfeit banknotes relies on numerical features extracted from the notes, such as variance, skewness, kurtosis, and entropy of their wavelet-transformed images. These features allow KNN to group similar notes together and differentiate fake ones from real ones.
- 🧩 Simplicity: KNN does not require complex mathematical modeling; it simply compares distances between points.
- 🔄 Flexibility: It works well with both linear and non-linear data distributions.
- 📊 Effectiveness in Low-Dimensional Space: Since the dataset consists of only four features, KNN can effectively classify notes without the need for complex feature engineering.

##### ⚖️ Choosing the Right K
Typically, choosing an optimal K involves cross-validation to find a balance between bias and variance. The parameter K (number of neighbors) plays a crucial role in KNN’s performance:
1. 🔎 Small K (e.g., 1-3):
- The model is highly sensitive to noise and may overfit to specific training examples.
- Can lead to misclassification if a fake note is very close to a genuine one.
2. 📉 Large K (e.g., 10+):
- The model becomes more stable but may smooth out important details.
- Can misclassify notes if fake banknotes are not well-represented in the training set.
- Typically, choosing an optimal K involves cross-validation to find a balance between bias and variance.

##### ⚠️ Limitations of KNN
Despite its advantages, KNN has some drawbacks when applied to fake banknote detection:
1. 🐢 Computational Cost:
- Since KNN stores all training samples, predicting a new note requires calculating distances to all points, which can be slow for large datasets.
2. 🧬 Sensitivity to Irrelevant Features:
- If irrelevant or highly correlated features are present, KNN’s performance may degrade.
3. ⚖️ Imbalanced Data Issues:
- If the dataset has significantly more genuine banknotes than fake ones, KNN might be biased toward predicting genuine notes.

##### 🚀 Improving KNN Performance
To enhance the effectiveness of KNN for detecting fake banknotes:
1. 📐 Feature Scaling:
- Since KNN relies on distances, features should be normalized (e.g., Min-Max scaling or Standardization) to prevent some features from dominating others.
2. 🧮 Dimensionality Reduction:
- Techniques like Principal Component Analysis (PCA) can remove redundant features, improving speed and accuracy.
3. ⚖️ Weighted Voting:
- Assigning greater importance to closer neighbors (distance-weighted KNN) can improve classification accuracy.
4. 🧠 Hybrid Approaches:
- Combining KNN with other classifiers (e.g., SVM, Decision Trees) or ensemble methods (e.g., Bagging, Boosting) can improve overall detection accuracy.

### 📊 Logistic Regression

#### 🏦 Logistic Regression in Detecting Fake Banknotes
Logistic Regression is a fundamental classification algorithm used to predict categorical outcomes. In the case of fake banknote detection, Logistic Regression helps classify a banknote as either genuine or counterfeit based on extracted numerical features such as variance, skewness, kurtosis, and entropy of wavelet-transformed images.

##### 🔎 How Logistic Regression Works
Unlike Linear Regression, which predicts continuous values, Logistic Regression is designed for binary classification problems (such as fake vs. genuine). It models the probability that a given input belongs to a particular class.

##### 🏆 Why Use Logistic Regression for Fake Banknote Detection?
Logistic Regression is a strong choice for fake banknote detection due to:
1. 📖 Interpretability:
- Unlike more complex models, Logistic Regression provides a clear probability score, helping understand how features influence classification.
2. ⚡ Efficiency:
- It is computationally inexpensive and works well for small to medium-sized datasets.
3. 📉 Robustness to Small Data:
- Unlike deep learning models that require massive datasets, Logistic Regression performs well even when labeled counterfeit banknote data is limited.
4. 📏 Well-Suited for Linearly Separable Data:
- If counterfeit banknotes have distinct statistical differences from real ones, Logistic Regression is highly effective.

##### 📌 Key Considerations in Logistic Regression

###### 📊 Feature Importance & Interpretation
1. 📈 Each coefficient 𝑤𝑖 indicates how strongly a feature affects the likelihood of a banknote being fake.
- If 𝑤𝑖 > 0, increasing that feature increases the probability of the banknote being counterfeit.
- If 𝑤𝑖 < 0, increasing that feature decreases the probability of the banknote being fake.
- Larger absolute values of 𝑤𝑖, indicate a stronger effect.
2. 📉 By analyzing these weights, we can determine which banknote characteristics are most predictive of counterfeits.

###### 🚨 Assumptions & Limitations
1. Linearity in Log-Odds:
- Logistic Regression assumes that each feature has a linear relationship with the log-odds of being fake, which may not always hold.
2. Sensitive to Outliers:
- Outliers can distort the decision boundary.
3. Imbalanced Classes:
- If there are far more genuine than fake banknotes, the model may be biased toward predicting "genuine" too often.

##### 🚀 Improving Logistic Regression for Fake Banknote Detection
To enhance performance, several techniques can be applied:
1. Feature Scaling:
- Standardizing features ensures that no single variable disproportionately influences predictions.
2. Regularization (L1/L2):
- L1 regularization (Lasso) can help select the most important features.
- L2 regularization (Ridge) prevents overfitting.
3. Handling Class Imbalance:
- If genuine banknotes outnumber fake ones, techniques like SMOTE (Synthetic Minority Over-sampling Technique) or class weighting can help balance the dataset.
4. Polynomial Features:
- If the relationship between banknote authenticity and features is nonlinear, adding polynomial terms can improve accuracy.

### Tree-Based Algorithms

The notebook [Tree.ipynb](Tree.ipynb) contains detailed implementations of various tree-based machine learning models. In this notebook, you will find techniques such as:
- **Classification Tree**: A decision tree algorithm for classification tasks.
```python
clftree = tree.DecisionTreeClassifier(max_depth = 3)
clftree.fit(X_train, y_train)
```
- **Bagging**: A technique that involves training multiple versions of a predictor on different subsets of the data and combining their predictions.
```python
clftree = tree.DecisionTreeClassifier(max_depth = 3)
bag_clf = BaggingClassifier(estimator = clftree,
                            n_estimators = 500,
                            bootstrap = True,
                            n_jobs = -1,
                            random_state = 42
                           )
bag_clf.fit(X_train, y_train)
```
- **Random Forest**: An ensemble method that uses multiple decision trees trained on different parts of the same dataset to improve classification accuracy.
```python
rf_clf = RandomForestClassifier(n_estimators = 500,
                                n_jobs = -1,
                                random_state = 42
                               )
rf_clf.fit(X_train, y_train)
```
- **Boosting**: A technique to combine the predictions of several base estimators to reduce bias and variance. It adjusts the weights of incorrectly classified instances so that subsequent classifiers focus more on difficult cases.
- **AdaBoost**: A specific type of boosting algorithm that combines multiple weak classifiers to create a strong classifier by focusing on misclassified instances.
```python
ada_clf = AdaBoostClassifier(learning_rate = 0.02,
                             n_estimators = 500
                            )
ada_clf.fit(X_train, y_train)
```
- **Gradient Boosting**: A boosting technique that builds models sequentially, with each new model correcting errors made by the previous ones, using gradient descent to minimize the loss function.
```python
xgb_clf = xgb.XGBClassifier(max_depth = 5,
                            n_estimators = 500,
                            learning_rate = 0.3,
                            n_jobs = -1
                           )
xgb_clf.fit(X_train, y_train)
```
- **XGBoost**: An advanced implementation of gradient boosting that includes regularization to prevent overfitting and improve model performance.
```python
gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)
```
- **Grid Search**: A method to perform hyperparameter optimization for machine learning models by exhaustively searching through a specified parameter grid.
```python
xgb_clf = xgb.XGBClassifier(n_estimators = 500,
                            learning_rate = 0.1,
                            random_state = 42
                           )
param_test1 = {
 'max_depth':range(3, 10, 2),
    'gamma' : [.1, .2, .3],
    'subsample':[.8, .9],
    'colsample_bytree':[.8, .9],
    'reg_alpha':[1e-2, .1, 1]
}
grid_search = GridSearchCV(xgb_clf,
                           param_test1,
                           n_jobs = -1,
                           cv = 5,
                           scoring = 'accuracy'
                          )
grid_search.fit(X_train, y_train)
```

#### Measuring the model performance

All tree based models were evaluated with below functions, which were created to speed up the process.

```python
# Define plot_confusion_matrix function
def plot_confusion_matrix(conf_matrix, labels = ['Not Fraud', 'Fraud'], title = 'Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot = True, cmap = 'Blues', fmt = 'g', 
                xticklabels=labels, 
                yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.show()

# Define Evaluate Model function
def EvaluateModel(y, x):
    # Model Evaluation
    print("Accuracy:", accuracy_score(y, x))
    
    # Print confusion matrix
    conf_matrix_test = confusion_matrix(y, x)
    print("Confusion Matrix:")
    print(conf_matrix_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix_test)
```
### 📌 SAS Flow

Encapsulating the ML models in the SAS Flow with Python Tools.

## Future Additions

This repository will be continually updated with more machine learning algorithms and techniques. Stay tuned for additions covering a wider range of models and methodologies.

## License

This repository is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE) file.

## Contribution

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request. Please ensure that your contributions are in line with the repository’s guidelines and coding standards.
