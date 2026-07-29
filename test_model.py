import joblib
import re


# ===========================
# Load trained model
# ===========================

model = joblib.load(
    "model/xgb_satd_model.pkl"
)

tfidf = joblib.load(
    "model/tfidf_vectorizer.pkl"
)

label_encoder = joblib.load(
    "model/label_encoder.pkl"
)



# ===========================
# Text preprocessing
# ===========================

def clean_comment(text):

    text = str(text)

    # Remove comment symbols
    text = re.sub(
        r'//|/\*|\*/|\*',
        '',
        text
    )

    # lowercase
    text = text.lower()

    # remove numbers
    text = re.sub(
        r'\d+',
        '',
        text
    )

    # remove special chars
    text = re.sub(
        r'[^a-z\s]',
        ' ',
        text
    )

    # remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()



# ===========================
# Prediction function
# ===========================

def predict_satd(comment):

    cleaned = clean_comment(comment)


    # Convert text to TF-IDF

    vector = tfidf.transform(
        [cleaned]
    )


    # Predict class

    prediction = model.predict(
        vector
    )


    # Convert number back to label

    label = label_encoder.inverse_transform(
        prediction
    )[0]


    # Probability

    probabilities = model.predict_proba(
        vector
    )


    confidence = max(probabilities[0])


    return {
        "comment": comment,
        "is_satd": label != "non_debt",
        "satd_type": None if label == "non_debt" else label,
        "confidence": round(float(confidence), 2)
    }



# ===========================
# Test Cases
# ===========================


test_comments = [

    "// TODO: Refactor this method because it has too many responsibilities",

    "// FIXME: This workaround is required because the legacy API is unstable",

    "// TODO: Optimize this database query. Currently it performs badly with large datasets",

    "// TODO: Add proper exception handling instead of catching generic Exception",

    "// FIXME: Remove this temporary solution after migrating to the new payment service",

    "// TODO: Add unit tests for this service before production release",

    "// FIXME: Missing integration tests for the authentication flow",

    "// TODO: Add API documentation for this endpoint",

    "// FIXME: Update the README with deployment instructions",

    "// TODO: Document why this algorithm uses this approach",

    "// TODO: Support multiple payment providers in the future",

    "// FIXME: This feature only supports single currency currently",

    "// TODO: Implement role-based access control for admin users",

    "// TODO: Replace this hardcoded configuration with environment variables",

    "// FIXME: Temporary database connection handling. Needs proper connection pooling",

    "// TODO: Improve logging and add monitoring support",

    "// HACK: We are bypassing validation here because of an old client issue",

    "// XXX: This implementation is slow but works for now",

    "// TODO: Add password complexity validation rules",

    "// TODO: Remove deprecated authentication method after all users migrate"

]


for comment in test_comments:

    result = predict_satd(comment)

    print("\n----------------------")

    print(result)