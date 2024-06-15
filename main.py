from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator

mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

some_digit = x.iloc[36000]
some_digit_image = some_digit.values.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()

print("Label:", y.iloc[36000])

x_train, x_test = x.iloc[:60000], x.iloc[60000:]
y_train, y_test = y.iloc[:60000], y.iloc[60000:]

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train.iloc[shuffle_index], y_train.iloc[shuffle_index]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)

clf = LogisticRegression(solver='lbfgs', max_iter=1000, tol=0.1, random_state=42)
clf.fit(x_train_scaled, y_train_2)

scaled_some_digit = scaler.transform([some_digit])
prediction = clf.predict(scaled_some_digit)
print("Prediction for the specific digit:", prediction)

test_accuracy = clf.score(x_test_scaled, y_test_2)
print("Test accuracy for detecting digit '2':", test_accuracy)

cross_val_scores = cross_val_score(clf, x_train_scaled, y_train_2, cv=3, scoring="accuracy")
print("Cross-validation scores:", cross_val_scores)
print("Mean cross-validation score:", cross_val_scores.mean())

class AlwaysNot2Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    
    def predict(self, X):
        return np.zeros((len(X),), dtype=bool)

not_2_clf = AlwaysNot2Classifier()


not_2_test_predictions = not_2_clf.predict(x_test_scaled)
test_accuracy_not_2 = (not_2_test_predictions == y_test_2).mean()
print("Test accuracy for AlwaysNot2Classifier:", test_accuracy_not_2)

cross_val_scores_not_2 = cross_val_score(not_2_clf, x_train_scaled, y_train_2, cv=3, scoring="accuracy")
print("Cross-validation scores for AlwaysNot2Classifier:", cross_val_scores_not_2)
print("Mean cross-validation score for AlwaysNot2Classifier:", cross_val_scores_not_2.mean())
