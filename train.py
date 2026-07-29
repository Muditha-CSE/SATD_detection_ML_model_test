import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.utils import resample

from xgboost import XGBClassifier



DATASET_PATH = "dataset/satd_dataset.csv"
MODEL_DIR = "model"



# ==================================
# Load Dataset
# ==================================

df = pd.read_csv(DATASET_PATH)


df = df.rename(
    columns={
        "text": "comment"
    }
)


df = df[
    [
        "comment",
        "classification"
    ]
].dropna()



# ==================================
# Remove Duplicate Comments
# ==================================

before = len(df)

df = df.drop_duplicates(
    subset=["comment"]
)


after = len(df)


print(
    f"Removed duplicates: {before-after}"
)



# ==================================
# Cleaning
# ==================================

def clean_comment(text):

    text = str(text)

    text = re.sub(
        r'//|/\*|\*/|\*',
        '',
        text
    )

    text = text.lower()

    text = re.sub(
        r'\d+',
        '',
        text
    )

    text = re.sub(
        r'[^a-z\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()



df["clean_comment"] = df["comment"].apply(
    clean_comment
)



# ==================================
# Encode Labels
# ==================================

X = df["clean_comment"]

y = df["classification"]


le = LabelEncoder()


y = le.fit_transform(y)



print("\nClass Mapping")

for i,c in enumerate(le.classes_):
    print(i,c)




# ==================================
# Train Test Split
# ==================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)



# ==================================
# TF-IDF
# ==================================

tfidf = TfidfVectorizer(

    lowercase=True,

    ngram_range=(1,2),

    max_features=20000

)



X_train = tfidf.fit_transform(
    X_train
)


X_test = tfidf.transform(
    X_test
)




print("\nOriginal Training Distribution")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)




# ==================================
# Training Balance
# ==================================

train_df = pd.DataFrame(
    {
        "index": range(len(y_train)),
        "label": y_train
    }
)



majority_class = (
    train_df["label"]
    .value_counts()
    .idxmax()
)



balanced_indices = []



for label in train_df["label"].unique():

    indexes = (
        train_df[
            train_df.label == label
        ]
        ["index"]
        .values
    )


    count = len(indexes)


    # Reduce majority class

    if label == majority_class:

        sampled = resample(

            indexes,

            replace=False,

            n_samples=5000,

            random_state=42

        )


    # Increase rare classes

    elif count < 500:

        sampled = resample(

            indexes,

            replace=True,

            n_samples=500,

            random_state=42

        )


    else:

        sampled = indexes



    balanced_indices.extend(
        sampled
    )




X_train = X_train[
    balanced_indices
]


y_train = y_train[
    balanced_indices
]




print("\nBalanced Training Distribution")


print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)




# ==================================
# Balanced Test Set
# ==================================

test_df = pd.DataFrame(

    {
        "index": range(len(y_test)),
        "label": y_test
    }

)



test_indices=[]



for label in test_df["label"].unique():

    indexes = (

        test_df[
            test_df.label == label
        ]
        ["index"]
        .values

    )


    sample_size = min(
        len(indexes),
        100
    )


    sampled = resample(

        indexes,

        replace=False,

        n_samples=sample_size,

        random_state=42

    )


    test_indices.extend(
        sampled
    )




X_test = X_test[
    test_indices
]


y_test = y_test[
    test_indices
]



print("\nBalanced Test Distribution")


print(
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)




# ==================================
# XGBoost Model
# ==================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=len(le.classes_),


    n_estimators=300,


    learning_rate=0.05,


    max_depth=3,


    min_child_weight=5,


    subsample=0.8,


    colsample_bytree=0.8,


    reg_alpha=1,


    reg_lambda=5,


    eval_metric="mlogloss",


    tree_method="hist",


    random_state=42

)




model.fit(

    X_train,

    y_train

)




# ==================================
# Evaluation
# ==================================

prediction = model.predict(
    X_test
)



accuracy = accuracy_score(

    y_test,

    prediction

)


macro_f1 = f1_score(

    y_test,

    prediction,

    average="macro"

)


weighted_f1 = f1_score(

    y_test,

    prediction,

    average="weighted"

)




print("\n======================")

print("Evaluation")

print("======================")



print(
    "Accuracy:",
    accuracy
)


print(
    "Macro F1:",
    macro_f1
)


print(
    "Weighted F1:",
    weighted_f1
)



print("\nClassification Report")


print(

    classification_report(

        y_test,

        prediction,

        target_names=le.classes_,

        digits=4,

        zero_division=0

    )

)



print("\nConfusion Matrix")


print(

    confusion_matrix(

        y_test,

        prediction

    )

)



# ==================================
# Save Model
# ==================================

os.makedirs(

    MODEL_DIR,

    exist_ok=True

)



joblib.dump(

    model,

    f"{MODEL_DIR}/xgb_satd_model.pkl"

)


joblib.dump(

    tfidf,

    f"{MODEL_DIR}/tfidf_vectorizer.pkl"

)


joblib.dump(

    le,

    f"{MODEL_DIR}/label_encoder.pkl"

)



print("\nTraining completed successfully.")