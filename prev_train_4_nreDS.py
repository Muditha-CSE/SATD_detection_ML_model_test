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



# =====================================
# Load Dataset
# =====================================

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



# =====================================
# Merge Classes
# =====================================

df["classification"] = df["classification"].replace(
    {
        "defect_debt": "functional_debt",
        "requirement_debt": "functional_debt",
        "documentation_debt": "functional_debt"
    }
)



print("\nDataset Distribution")

print(
    df["classification"]
    .value_counts()
)



# =====================================
# Text Cleaning
# =====================================

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


    # keep programming words
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



# =====================================
# Keyword Features
# =====================================

satd_keywords = [

    # general SATD
    "todo",
    "fixme",
    "hack",
    "temporary",
    "workaround",
    "later",
    "deprecated",
    "obsolete",
    "cleanup",


    # design debt
    "refactor",
    "complex",
    "duplicate",
    "bad design",
    "poor design",


    # functional debt
    "bug",
    "error",
    "exception",
    "missing",
    "not implemented",
    "need support",
    "should support",
    "requirement",
    "documentation",
    "document",


    # test debt
    "test",
    "unit test",
    "coverage",
    "testing"

]



def keyword_features(text):

    result = []

    for word in satd_keywords:

        if word in text:
            result.append(1)

        else:
            result.append(0)


    return result



keyword_matrix = csr_matrix(

    df["clean_comment"]
    .apply(keyword_features)
    .tolist()

)



# =====================================
# Labels
# =====================================

X = df["clean_comment"]


encoder = LabelEncoder()


y = encoder.fit_transform(
    df["classification"]
)



print("\nClass Mapping")

for index, label in enumerate(encoder.classes_):

    print(index, "=", label)



# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test, kw_train, kw_test = train_test_split(

    X,
    y,
    keyword_matrix,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# =====================================
# Word TF-IDF
# =====================================

word_vectorizer = TfidfVectorizer(

    analyzer="word",

    lowercase=True,

    stop_words=None,

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



# =====================================
# Character TF-IDF
# =====================================

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



# =====================================
# Combine Features
# =====================================

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



print("\nTraining Distribution")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)



# =====================================
# Train SVM
# =====================================

model = LinearSVC(

    C=1.0,

    class_weight="balanced",

    max_iter=30000,

    random_state=42

)



model.fit(

    X_train_final,

    y_train

)



# =====================================
# Evaluation
# =====================================

pred = model.predict(
    X_test_final
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

        target_names=encoder.classes_,

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



# =====================================
# Save Model
# =====================================

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
classification
non_debt            35522
code/design_debt     2262
functional_debt       950
test_debt              81
Name: count, dtype: int64

Class Mapping
0 = code/design_debt
1 = functional_debt
2 = non_debt
3 = test_debt

Training Distribution
0     1809
1      760
2    28418
3       65
Name: count, dtype: int64

Accuracy:
0.9585211902614968

Macro F1:
0.6957038900945359

Weighted F1:
0.9569512184123872

Classification Report
                  precision    recall  f1-score   support

code/design_debt     0.7181    0.7086    0.7133       453
 functional_debt     0.5608    0.4368    0.4911       190
        non_debt     0.9821    0.9894    0.9858      7104
       test_debt     0.7273    0.5000    0.5926        16

        accuracy                         0.9585      7763
       macro avg     0.7471    0.6587    0.6957      7763
    weighted avg     0.9559    0.9585    0.9570      7763


Confusion Matrix
[[ 321   42   89    1]
 [  71   83   34    2]
 [  54   21 7029    0]
 [   1    2    5    8]]

Training completed successfully."""