import os
import re
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

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



# =========================================
# Load Dataset
# =========================================

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


df = df.drop_duplicates(
    subset=["comment"]
)



# =========================================
# Cleaning
# =========================================

def clean_comment(text):

    text = str(text).lower()

    # remove urls
    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    # remove comment symbols
    text = re.sub(
        r"//|/\*|\*/|\*",
        " ",
        text
    )

    # keep words, numbers, underscore
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



df["clean_comment"] = df["comment"].apply(
    clean_comment
)



# =========================================
# Reduce non_debt dominance
# =========================================

non_debt = df[
    df["classification"] == "non_debt"
]


satd = df[
    df["classification"] != "non_debt"
]


# keep enough non debt examples
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


print("\nDataset distribution")

print(
    df["classification"]
    .value_counts()
)



# =========================================
# SATD keyword features
# =========================================

satd_keywords = [

    "todo",
    "fixme",
    "hack",
    "temporary",
    "workaround",
    "technical debt",
    "quick fix",
    "remove later",
    "need to refactor",
    "should improve",
    "not implemented",
    "missing support",
    "cleanup",
    "obsolete",
    "deprecated",
    "bad design",
    "poor implementation",
    "later"

]



def keyword_features(text):

    result = []

    for keyword in satd_keywords:

        if keyword in text:
            result.append(1)
        else:
            result.append(0)

    return result



keyword_matrix = csr_matrix(

    df["clean_comment"]
    .apply(keyword_features)
    .tolist()

)



# =========================================
# Labels
# =========================================

X = df["clean_comment"]


encoder = LabelEncoder()


y = encoder.fit_transform(
    df["classification"]
)



# =========================================
# Split Dataset
# =========================================

X_train, X_test, y_train, y_test, kw_train, kw_test = train_test_split(

    X,
    y,
    keyword_matrix,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# =========================================
# Word TF-IDF
# =========================================

word_vectorizer = TfidfVectorizer(

    analyzer="word",

    lowercase=True,

    ngram_range=(1,3),

    max_features=100000,

    sublinear_tf=True,

    min_df=2

)



X_train_word = word_vectorizer.fit_transform(
    X_train
)


X_test_word = word_vectorizer.transform(
    X_test
)



# =========================================
# Character TF-IDF
# =========================================

char_vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(3,5),

    max_features=100000,

    sublinear_tf=True,

    min_df=2

)



X_train_char = char_vectorizer.fit_transform(
    X_train
)


X_test_char = char_vectorizer.transform(
    X_test
)



# =========================================
# Combine Features
# =========================================

X_train_final = hstack(
    [
        X_train_word,
        X_train_char,
        kw_train
    ]
)


X_test_final = hstack(
    [
        X_test_word,
        X_test_char,
        kw_test
    ]
)



print("\nTraining distribution")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)



# =========================================
# Linear SVM
# =========================================

model = LinearSVC(

    C=0.5,

    class_weight="balanced",

    max_iter=30000,

    random_state=42

)



model.fit(

    X_train_final,

    y_train

)



# =========================================
# Evaluation
# =========================================

prediction = model.predict(
    X_test_final
)



print("\nAccuracy:")

print(
    accuracy_score(
        y_test,
        prediction
    )
)



print("\nMacro F1:")

print(
    f1_score(
        y_test,
        prediction,
        average="macro"
    )
)



print("\nWeighted F1:")

print(
    f1_score(
        y_test,
        prediction,
        average="weighted"
    )
)



print("\nClassification Report")


print(

    classification_report(

        y_test,

        prediction,

        target_names=encoder.classes_,

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



# =========================================
# Save Models
# =========================================

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

    word_vectorizer,

    os.path.join(
        MODEL_DIR,
        "word_tfidf.pkl"
    )

)



joblib.dump(

    char_vectorizer,

    os.path.join(
        MODEL_DIR,
        "char_tfidf.pkl"
    )

)



joblib.dump(

    encoder,

    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )

)



print("\nTraining completed successfully.")


""""
Accuracy:
0.8950733358405416

Macro F1:
0.6144776489747737

Weighted F1:
0.8890022563586534

Classification Report
                    precision    recall  f1-score   support

  code/design_debt     0.7467    0.7417    0.7442       453
       defect_debt     0.5128    0.2857    0.3670        70
documentation_debt     1.0000    0.4000    0.5714        10
          non_debt     0.9530    0.9825    0.9675      2000
  requirement_debt     0.5109    0.4273    0.4653       110
         test_debt     0.6667    0.5000    0.5714        16

          accuracy                         0.8951      2659
         macro avg     0.7317    0.5562    0.6145      2659
      weighted avg     0.8864    0.8951    0.8890      2659


Confusion Matrix
[[ 336   11    0   72   33    1]
 [  27   20    0   15    6    2]
 [   5    1    4    0    0    0]
 [  26    2    0 1965    6    1]
 [  50    4    0    9   47    0]
 [   6    1    0    1    0    8]]

Training completed successfully."""