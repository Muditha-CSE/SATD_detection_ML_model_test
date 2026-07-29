import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


DATASET_PATH = "dataset/satd_dataset.csv"
MODEL_DIR = "model"


# Load dataset
df = pd.read_csv(DATASET_PATH)

df = df.rename(columns={"text": "comment"})

df = df[
    ["comment", "classification"]
].dropna().drop_duplicates(subset=["comment"])



# Improved cleaning
def clean_comment(text):

    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove comment symbols but keep words
    text = re.sub(
        r"//|/\*|\*/|\*",
        " ",
        text
    )

    # keep important SATD indicators
    text = re.sub(
        r"[^a-z0-9_\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



df["clean_comment"] = df["comment"].apply(clean_comment)



# Reduce non_debt dominance

non_debt = df[
    df["classification"] == "non_debt"
]

satd = df[
    df["classification"] != "non_debt"
]


# Keep only part of non debt samples
non_debt_sample = non_debt.sample(
    n=10000,
    random_state=42
)


df = pd.concat(
    [
        non_debt_sample,
        satd
    ],
    ignore_index=True
)


print("\nDataset after balancing:")
print(
    df["classification"].value_counts()
)



X = df["clean_comment"]


# Encode labels
le = LabelEncoder()

y = le.fit_transform(
    df["classification"]
)



# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# TF-IDF

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),
    max_features=80000,
    sublinear_tf=True,
    min_df=2
)


X_train = tfidf.fit_transform(
    X_train
)

X_test = tfidf.transform(
    X_test
)



print("\nTraining distribution:")
print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)



# Linear SVM

model = LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=20000,
    random_state=42
)



model.fit(
    X_train,
    y_train
)



# Prediction

pred = model.predict(
    X_test
)



print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        pred
    )
)


print("\nMacro F1:")
print(
    f1_score(
        y_test,
        pred,
        average="macro"
    )
)


print("\nWeighted F1:")
print(
    f1_score(
        y_test,
        pred,
        average="weighted"
    )
)



print("\nClassification Report")

print(
    classification_report(
        y_test,
        pred,
        target_names=le.classes_,
        digits=4,
        zero_division=0
    )
)



print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        pred
    )
)



# Save models

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "svm_satd_model.pkl"
    )
)


joblib.dump(
    tfidf,
    os.path.join(
        MODEL_DIR,
        "tfidf_vectorizer.pkl"
    )
)


joblib.dump(
    le,
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)


print("\nTraining completed successfully.")