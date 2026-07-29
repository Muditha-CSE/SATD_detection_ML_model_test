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
# Text Cleaning
# =====================================

def clean_comment(text):

    text = str(text).lower()

    # remove URLs
    text = re.sub(
        r"http\S+",
        " ",
        text
    )

    # remove comment syntax
    text = re.sub(
        r"//|/\*|\*/|\*",
        " ",
        text
    )

    # keep programming words and numbers
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
# Reduce Non Debt Dominance
# =====================================

non_debt = df[
    df["classification"] == "non_debt"
]


satd = df[
    df["classification"] != "non_debt"
]


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



# =====================================
# SATD Phrase Features
# =====================================

satd_patterns = [

    "todo",
    "fixme",
    "hack",
    "temporary",
    "workaround",
    "technical debt",
    "quick fix",
    "need to refactor",
    "should be improved",
    "remove later",
    "not implemented",
    "missing support",
    "later",
    "cleanup",
    "obsolete",
    "deprecated",
    "bad design",
    "poor implementation"

]


def keyword_features(text):

    features = []

    for phrase in satd_patterns:

        if phrase in text:
            features.append(1)
        else:
            features.append(0)

    return features



keyword_matrix = csr_matrix(
    df["clean_comment"]
    .apply(keyword_features)
    .tolist()
)



# =====================================
# Label Encoding
# =====================================

X = df["clean_comment"]


label_encoder = LabelEncoder()


y = label_encoder.fit_transform(
    df["classification"]
)



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



print("\nTraining distribution")

print(
    pd.Series(y_train)
    .value_counts()
    .sort_index()
)



# =====================================
# Custom Class Weights
# =====================================

class_weights = {


    # code/design debt
    0: 1.5,


    # defect debt
    1: 5.0,


    # documentation debt
    2: 8.0,


    # non debt
    3: 1.0,


    # requirement debt
    4: 5.0,


    # test debt
    5: 6.0

}



# =====================================
# Train Linear SVM
# =====================================

model = LinearSVC(

    C=1.0,

    class_weight=class_weights,

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

        target_names=label_encoder.classes_,

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



# =====================================
# Save Models
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

    label_encoder,

    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )

)



print("\nTraining completed successfully.")

"""Dataset distribution
classification
non_debt              10000
code/design_debt       2262
requirement_debt        550
defect_debt             351
test_debt                81
documentation_debt       49
Name: count, dtype: int64

Training distribution
0    1809
1     281
2      39
3    8000
4     440
5      65
Name: count, dtype: int64

Accuracy:
0.8935690109063558

Macro F1:
0.5843859929013852

Weighted F1:
0.885711543788439

Classification Report
                    precision    recall  f1-score   support

  code/design_debt     0.7364    0.7461    0.7412       453
       defect_debt     0.4444    0.2286    0.3019        70
documentation_debt     1.0000    0.4000    0.5714        10
          non_debt     0.9507    0.9840    0.9671      2000
  requirement_debt     0.5500    0.4000    0.4632       110
         test_debt     0.6000    0.3750    0.4615        16

          accuracy                         0.8936      2659
         macro avg     0.7136    0.5223    0.5844      2659
      weighted avg     0.8824    0.8936    0.8857      2659


Confusion Matrix
[[ 338   13    0   75   26    1]
 [  31   16    0   17    4    2]
 [   6    0    4    0    0    0]
 [  25    1    0 1968    5    1]
 [  53    5    0    8   44    0]
 [   6    1    0    2    1    6]]

Training completed successfully."""