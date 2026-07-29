import os
import re
import joblib

from scipy.sparse import hstack, csr_matrix



MODEL_DIR = "model"



# =====================================================
# Load Models
# =====================================================


model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "svm_satd_model.pkl"
    )
)


word_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "word_tfidf.pkl"
    )
)


char_vectorizer = joblib.load(
    os.path.join(
        MODEL_DIR,
        "char_tfidf.pkl"
    )
)


encoder = joblib.load(
    os.path.join(
        MODEL_DIR,
        "label_encoder.pkl"
    )
)


satd_features = joblib.load(
    os.path.join(
        MODEL_DIR,
        "satd_features.pkl"
    )
)





# =====================================================
# Cleaning
# =====================================================


def clean_comment(text):


    text=str(text).lower()


    text=re.sub(
        r"http\S+",
        " ",
        text
    )


    text=re.sub(
        r"//|/\*|\*/|\*",
        " ",
        text
    )


    text=re.sub(
        r"[^a-z0-9_\s]",
        " ",
        text
    )


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()






# =====================================================
# SAME SATD FEATURE EXTRACTION
# =====================================================


def extract_satd_features(text):


    features=[]



    # Weighted scores

    for category,keywords in satd_features.items():


        score=0


        for phrase,weight in keywords.items():


            if phrase in text:


                score += weight * 5



        features.append(score)





    # Binary indicators

    for category,keywords in satd_features.items():


        found=0


        for phrase in keywords:


            if phrase in text:

                found=1

                break



        features.append(found)



    return features







# =====================================================
# Prediction
# =====================================================


def predict(comment):


    cleaned = clean_comment(comment)



    word_features = word_vectorizer.transform(
        [cleaned]
    )



    char_features = char_vectorizer.transform(
        [cleaned]
    )



    satd_feature = csr_matrix(
        [
            extract_satd_features(cleaned)
        ]
    )



    final_features = hstack(

        [

            word_features,

            char_features,

            satd_feature

        ]

    )



    print(
        "Feature size:",
        final_features.shape[1]
    )


    print(
        "Model expects:",
        model.n_features_in_
    )



    prediction = model.predict(
        final_features
    )



    label = encoder.inverse_transform(
        prediction
    )



    # SVM confidence

    decision = model.decision_function(
        final_features
    )


    confidence = max(
        decision[0]
    )



    return label[0], confidence






# =====================================================
# Test Comments
# =====================================================


test_comments=[



"""
TODO this class is too complex.
Need refactoring because the design is difficult to maintain.
""",



"""
FIXME this causes a NullPointerException.
Validation logic is broken.
""",



"""
Support for multiple payment providers is not implemented yet.
This feature should be implemented later.
""",



"""
Fix typo in API documentation.
Update README and document missing information.
""",



"""
TODO add unit tests for this function.
Current code has no test coverage.
""",



"""
Calculate the total payment amount and return the response.
""",



"""
This method contains duplicate code and poor design.
Refactor this implementation.
""",



"""
Integration testing is missing for the payment service.
Coverage should be improved.
""",



"""
The application crashes when database connection fails.
Exception handling needs improvement.
""",



"""
The system does not support Google login.
OAuth authentication needs to be added.
""",



"""
The API documentation is outdated.
Update the developer guide with new endpoints.
""",



"""
This class has memory leaks and unnecessary complexity.
Cleanup is required.
""",



"""
The current tests are flaky and fail randomly.
Investigate unstable test cases.
""",



"""
Create a new user account and save the details.
""",



"""
Initialize the database connection pool.
"""

]







# =====================================================
# Run
# =====================================================


print("\nSATD RESULTS")
print("="*60)



for comment in test_comments:


    label,score = predict(comment)



    print("\nComment:")
    print(comment.strip())


    print("\nPrediction:")
    print(label)


    print("\nConfidence:")
    print(
        round(score,4)
    )


    print("-"*60)