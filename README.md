# Machine Learning Algorithms

Welcome to the Machine Learning Algorithms repository! This repository is dedicated to providing comprehensive examples and implementations of various machine learning algorithms. All content in this repository is licensed under the MIT License.

## Contents

### KNN

K-Nearest Neighbors (KNN) is a simple yet effective machine learning algorithm used for classification and regression tasks. In the context of detecting fake banknotes, KNN helps classify new banknotes as either genuine or counterfeit based on their similarity to known examples.

KNN is a non-parametric, instance-based learning algorithm, meaning it does not make strong assumptions about the data distribution and instead stores all training data to make predictions when needed. The core idea behind KNN is as follows:

Distance Measurement:

When given a new banknote, KNN computes its distance from all training samples using a distance metric, typically Euclidean distance.
Selecting K Neighbors:

The algorithm selects the K nearest data points based on the calculated distances.
Voting Mechanism:

In classification tasks (like fake vs. genuine banknotes), KNN assigns the class label based on the majority vote among the K nearest neighbors.
Final Classification:

The new banknote is classified as genuine or counterfeit depending on which class is more frequent among its neighbors.

Detecting counterfeit banknotes relies on numerical features extracted from the notes, such as variance, skewness, kurtosis, and entropy of their wavelet-transformed images. These features allow KNN to group similar notes together and differentiate fake ones from real ones.

Simplicity: KNN does not require complex mathematical modeling; it simply compares distances between points.
Flexibility: It works well with both linear and non-linear data distributions.
Effectiveness in Low-Dimensional Space: Since the dataset consists of only four features, KNN can effectively classify notes without the need for complex feature engineering.

### Logistic Regression

...

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

## Future Additions

This repository will be continually updated with more machine learning algorithms and techniques. Stay tuned for additions covering a wider range of models and methodologies.

## License

This repository is licensed under the MIT License. For more details, please refer to the [LICENSE](LICENSE) file.

## Contribution

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request. Please ensure that your contributions are in line with the repository’s guidelines and coding standards.
