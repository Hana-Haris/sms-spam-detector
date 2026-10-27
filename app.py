import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Website settings
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Website design
st.markdown("""
<style>
.stApp {
    background-color: #f2f6fc;
}
.title {
    text-align: center;
    color: #173b68;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #536579;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_resource
def train_model():
    data = pd.read_csv(
        "data/SMSSpamCollection",
        sep="\t",
        names=["label", "message"],
        encoding="utf-8"
    )

    # Convert spam and ham into numbers
    data["label"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Separate messages and labels
    X = data["message"]
    y = data["label"]

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Convert text into numerical features
    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Train SVM model
    model = LinearSVC()
    model.fit(X_train, y_train)

    # Calculate test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return vectorizer, model, accuracy


# Train the model
try:
    vectorizer, model, accuracy = train_model()
except Exception as e:
    st.error("Could not load the dataset. Check the data folder.")
    st.stop()

# ================= FRONTEND DESIGN =================

st.markdown("""
<style>
/* Main page */
.stApp {
    background: #F4F7FC;
    color: #172B4D;
}

.block-container {
    max-width: 820px;
    padding: 2.5rem 1.5rem 2rem;
}

/* Hide default decoration */
#MainMenu, footer {
    visibility: hidden;
}

/* Header card */
.header-card {
    background: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 18px;
    padding: 25px 28px;
    margin-bottom: 24px;
}

.brand {
    color: #2855B7;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.main-title {
    color: #172B4D;
    font-size: 34px;
    font-weight: 750;
    line-height: 1.25;
    margin: 0;
}

.description {
    color: #718096;
    font-size: 15px;
    margin-top: 10px;
}

/* Input label */
.stTextArea label {
    color: #243B60;
    font-weight: 600;
    font-size: 15px;
}

/* Text input */
.stTextArea textarea {
    background: #FFFFFF;
    color: #243B60;
    border: 1px solid #DCE4F0;
    border-radius: 12px;
    padding: 14px;
    font-size: 15px;
    line-height: 1.6;
    transition: border-color 0.2s;
}

.stTextArea textarea:focus {
    border-color: #4779D5;
    box-shadow: 0 0 0 2px #E5EDFF;
}

/* Analyze button */
.stFormSubmitButton button {
    background: #2855B7;
    color: #FFFFFF;
    border: 1px solid #2855B7;
    border-radius: 10px;
    min-height: 46px;
    font-size: 15px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stFormSubmitButton button:hover {
    background: #1D438F;
    border-color: #1D438F;
    color: #FFFFFF;
}

/* Results */
div[data-testid="stAlert"] {
    border-radius: 12px;
    border-width: 1px;
    padding: 16px;
}

/* Model information */
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E3E9F2;
    border-radius: 12px;
    margin-top: 22px;
}

div[data-testid="stMetric"] {
    background: #F7F9FD;
    border: 1px solid #E7ECF4;
    border-radius: 10px;
    padding: 14px;
}

/* Footer */
.footer {
    border-top: 1px solid #E3E9F2;
    margin-top: 32px;
    padding-top: 18px;
    text-align: center;
    color: #8A97AB;
    font-size: 12px;
}

/* Small screen adjustments */
@media (max-width: 600px) {
    .block-container {
        padding: 1.2rem 1rem;
    }

    .header-card {
        padding: 21px 18px;
    }

    .main-title {
        font-size: 28px;
    }
}
</style>
""", unsafe_allow_html=True)


# ================= HEADER =================

st.markdown("""
<div class="header-card">
    <div class="brand">◈ &nbsp; SMS SHIELD</div>
    <div class="main-title">Check your message.</div>
    <div class="description">
        Identify suspicious SMS messages using
        machine learning.
    </div>
</div>
""", unsafe_allow_html=True)


# ================= MESSAGE INPUT =================

st.markdown("#### Message analysis")
st.caption("Paste the SMS you want to check below.")

with st.form("message_form"):
    message = st.text_area(
        "Your message",
        placeholder=(
            "Example: Congratulations! "
            "You have won a free prize..."
        ),
        height=145,
        label_visibility="visible"
    )

    submitted = st.form_submit_button(
        "⌕   Analyze Message",
        use_container_width=True
    )


# ================= PREDICTION =================

if submitted:
    if not message.strip():
        st.warning("Please enter a message before analyzing.")
    else:
        message_data = vectorizer.transform([message])
        result = model.predict(message_data)[0]

        st.markdown("#### Analysis result")

        if result == 1:
            st.error(
                "⚠️  Potential spam detected\n\n"
                "This message resembles spam messages "
                "in the training dataset."
            )
        else:
            st.success(
                "✓  Message classified as not spam\n\n"
                "This message resembles legitimate messages "
                "in the training dataset."
            )

        st.caption(
            "Machine learning predictions may be incorrect. "
            "Avoid clicking suspicious links."
        )


# ================= MODEL DETAILS =================

with st.expander("Model details and performance"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Classification algorithm**")
        st.write("Linear Support Vector Machine")

    with col2:
        st.markdown("**Text representation**")
        st.write("TF-IDF")

    
    st.markdown(
    f"""
    <div style="
        background: #F7F9FD;
        border: 1px solid #E3E9F2;
        border-radius: 10px;
        padding: 16px;
        margin-top: 15px;
        text-align: center;
    ">
        <div style="color: #718096; font-size: 14px;">
            Test Accuracy
        </div>
        <div style="
            color: #2855B7;
            font-size: 30px;
            font-weight: bold;
        ">
            {accuracy * 100:.2f}%
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    st.caption(
        "Accuracy is calculated using a held-out test dataset."
    )


# ================= FOOTER =================

st.markdown("""
<div class="footer">
    SMS SHIELD &nbsp; · &nbsp;
    Machine Learning Lab Microproject
</div>
""", unsafe_allow_html=True)