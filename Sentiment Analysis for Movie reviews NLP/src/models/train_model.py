import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
## read dataset

df = pd.read_csv("../../data/interim/cleaned_text.csv")
X = df["review"]
Y = df["sentiment"]

## train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

## training MultinomialNB

MultiNB_TF_IDF_pipline = Pipeline(
    [
        ("TF_IFD", TfidfVectorizer(lowercase=False, strip_accents=None)),
        ("Multi_NB", MultinomialNB()),
    ]
)
MultiNB_TF_IDF_pipline.fit(X_train, Y_train)

##check our model performance and overfitting

y_pred = MultiNB_TF_IDF_pipline.predict(X_train)
acc = accuracy_score(Y_train, y_pred)
print(f"Model accuracy on train data = {acc}")

scores = cross_val_score(
    MultiNB_TF_IDF_pipline, X_train, Y_train, scoring="accuracy", cv=5
)
print(f"Mean CV Accuracy: {scores.mean():.2f}")
print(f"Standard deviation CV Accuracy: {scores.std():.6f}")

## Learning curve

train_sizes, train_scores, test_scores = learning_curve(
    estimator=MultiNB_TF_IDF_pipline,
    X=X_train,
    y=Y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

# Calculate means and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, "o-", color="blue", label="Training score")
plt.plot(train_sizes, test_mean, "o-", color="green", label="Cross-validation score")
plt.title("Learning Curve for MultinomialNB")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("../../reports/figures/learning_curve")
plt.show()


## check performance on test data

y_pred = MultiNB_TF_IDF_pipline.predict(X_test)
acc = accuracy_score(Y_test, y_pred)
print(f"Model accuracy on test data = {acc}")
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))

## save trained model

joblib.dump(MultiNB_TF_IDF_pipline, "../../models/multinb_tfidf_pipeline.pkl")
