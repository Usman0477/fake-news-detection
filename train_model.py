# Import Path to handle file and folder paths
from pathlib import Path

# Import json to save model evaluation results in JSON format
import json

# Import re for regular expression operations (text cleaning)
import re

# Import pandas to read and manipulate CSV datasets
import pandas as pd

# Import joblib to save the trained machine learning model
import joblib


# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import Pipeline to combine multiple machine learning steps
# FeatureUnion to combine word-based and character-based text features
from sklearn.pipeline import Pipeline, FeatureUnion

# Import TfidfVectorizer to convert text into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Logistic Regression as our machine learning classification algorithm
from sklearn.linear_model import LogisticRegression

# Import evaluation metrics to measure model performance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

# Get the directory where the current Python file is located
ROOT = Path(__file__).resolve().parent

# Define the folder containing our dataset CSV files
DATA_DIR = ROOT / "data"

# Define the folder where our trained model and evaluation results will be saved
MODEL_DIR = ROOT / "models"

# Define the complete path where the trained machine learning model will be stored
MODEL_PATH = MODEL_DIR / "fake_news_model.joblib"


# ============================================================
# 2. TEXT PREPROCESSING FUNCTION
# ============================================================

def clean_text(text):
    """
    This function performs basic text cleaning.

    It removes HTML tags and unnecessary whitespace.
    TF-IDF will handle tokenization and lowercasing later.
    """

    # Convert missing values into empty strings
    # Otherwise, convert the input into a string
    text = str(text) if pd.notna(text) else ""

    # Remove HTML tags from the text
    # Example: <p>Hello</p> becomes Hello
    text = re.sub(r"<[^>]+>", " ", text)

    # Replace multiple spaces, tabs, or newlines with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove whitespace from the beginning and end of the text
    return text.strip()


# ============================================================
# 3. LOAD AND PREPARE THE ISOT DATASET
# ============================================================

def load_isot_data():

    # Define the path of the fake news CSV file
    fake_path = DATA_DIR / "Fake.csv"

    # Define the path of the real news CSV file
    true_path = DATA_DIR / "True.csv"

    # Check whether both dataset files exist
    if not fake_path.exists() or not true_path.exists():

        # Raise an error if either dataset file is missing
        raise FileNotFoundError(
            "Could not find data/Fake.csv and data/True.csv. "
            "Put both downloaded ISOT CSV files inside the project's data folder."
        )

    # Read the fake news dataset using pandas
    fake = pd.read_csv(fake_path)

    # Read the real news dataset using pandas
    real = pd.read_csv(true_path)

    # ISOT dataset normally contains:
    # title, text, subject, and date columns.
    #
    # We only use title and text because subject and date
    # can introduce unwanted dataset/source clues.

    # Assign labels to fake and real news datasets
    # Fake = 0
    # Real = 1
    for frame, label in ((fake, 0), (real, 1)):

        # Check whether the required columns exist
        for col in ("title", "text"):

            # Raise an error if a required column is missing
            if col not in frame.columns:
                raise ValueError(
                    f"Expected a '{col}' column in the CSV. Found: {list(frame.columns)}"
                )

        # Add a label column to identify the news category
        frame["label"] = label

        # Combine the news title and article text into one column
        # This gives the model more textual information for classification
        frame["content"] = (
            frame["title"].fillna("").astype(str) + ". " +
            frame["text"].fillna("").astype(str)
        ).map(clean_text)

    # Combine fake and real datasets into one DataFrame
    # Only keep the content and label columns
    data = pd.concat(
        [fake[["content", "label"]], real[["content", "label"]]],
        ignore_index=True
    )

    # Remove articles having fewer than 30 characters
    # Very short texts may not contain enough information
    data = data[data["content"].str.len() >= 30].copy()

    # Remove duplicate news articles
    data.drop_duplicates(subset=["content"], inplace=True)

    # Randomly shuffle the dataset
    # random_state=42 ensures reproducible results
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Check that the dataset contains both fake and real labels
    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both fake and real examples.")

    # Return the cleaned and prepared dataset
    return data


# ============================================================
# 4. MAIN FUNCTION - MODEL TRAINING
# ============================================================

def main():

    # Create the models directory if it does not already exist
    MODEL_DIR.mkdir(exist_ok=True)

    # Display a message before loading the dataset
    print("Loading ISOT dataset...")

    # Load and prepare the ISOT fake news dataset
    data = load_isot_data()

    # Store article text in X (input features)
    X = data["content"]

    # Store labels in y (target/output variable)
    y = data["label"]

    # Display the total number of usable articles
    print(f"Usable articles: {len(data):,}")

    # Display the distribution of fake and real news labels
    print("Labels (0=fake, 1=real):")

    print(y.value_counts().sort_index().rename(index={0: "Fake", 1: "Real"}))

    # ========================================================
    # 5. SPLIT DATASET INTO TRAINING AND TESTING DATA
    # ========================================================

    # Split the dataset into:
    # 80% training data
    # 20% testing data
    #
    # random_state=42 ensures the same split every time
    # stratify=y maintains the proportion of fake and real news
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ========================================================
    # 6. FEATURE EXTRACTION USING TF-IDF
    # ========================================================

    # FeatureUnion combines multiple feature extraction methods
    #
    # We use two types of TF-IDF:
    #
    # 1. Word-level TF-IDF
    # 2. Character-level TF-IDF
    #
    # Combining them allows the model to learn different
    # patterns from news articles.

    features = FeatureUnion([

        # ----------------------------------------------------
        # WORD-LEVEL TF-IDF
        # ----------------------------------------------------

        ("word_tfidf", TfidfVectorizer(

            # Extract individual words and two-word combinations
            # Example:
            # "breaking news"
            # "breaking"
            # "news"
            ngram_range=(1, 2),

            # Ignore words appearing in fewer than 2 documents
            min_df=2,

            # Ignore terms appearing in more than 98% of documents
            max_df=0.98,

            # Limit the maximum number of word features
            max_features=250_000,

            # Apply sublinear term frequency scaling
            sublinear_tf=True,

            # Normalize different accent representations
            strip_accents="unicode"
        )),

        # ----------------------------------------------------
        # CHARACTER-LEVEL TF-IDF
        # ----------------------------------------------------

        ("char_tfidf", TfidfVectorizer(

            # Extract character sequences between 3 and 5 characters
            analyzer="char_wb",
            ngram_range=(3, 5),

            # Ignore character features appearing in fewer than 2 documents
            min_df=2,

            # Limit the maximum number of character features
            max_features=200_000,

            # Apply sublinear term frequency scaling
            sublinear_tf=True
        ))
    ])

    # ========================================================
    # 7. BUILD THE MACHINE LEARNING PIPELINE
    # ========================================================

    # Pipeline combines feature extraction and classification
    #
    # First:
    # Convert text into numerical TF-IDF features
    #
    # Second:
    # Use Logistic Regression to classify news as fake or real

    model = Pipeline([

        # Feature extraction stage
        ("features", features),

        # Classification stage
        ("classifier", LogisticRegression(

            # Regularization strength parameter
            C=4.0,

            # Maximum number of iterations allowed during training
            max_iter=1200,

            # Optimization solver used for Logistic Regression
            solver="liblinear"
        ))
    ])

    # ========================================================
    # 8. TRAIN THE MACHINE LEARNING MODEL
    # ========================================================

    # Display a training message
    print("\nTraining model... This may take a few minutes.")

    # Train the complete pipeline using training data
    model.fit(X_train, y_train)

    # ========================================================
    # 9. MAKE PREDICTIONS ON TEST DATA
    # ========================================================

    # Use the trained model to predict labels for unseen test articles
    predictions = model.predict(X_test)

    # ========================================================
    # 10. MODEL EVALUATION
    # ========================================================

    # Accuracy:
    # Measures the percentage of total correct predictions
    accuracy = accuracy_score(y_test, predictions)

    # Precision:
    # Measures how many predicted positive cases were actually positive
    precision = precision_score(y_test, predictions, zero_division=0)

    # Recall:
    # Measures how many actual positive cases were correctly identified
    recall = recall_score(y_test, predictions, zero_division=0)

    # F1-score:
    # Harmonic mean of precision and recall
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Display the evaluation results
    print("\n===== HELD-OUT TEST RESULTS =====")

    # Display accuracy
    print(f"Accuracy : {accuracy:.4f}")

    # Display precision
    print(f"Precision: {precision:.4f}")

    # Display recall
    print(f"Recall   : {recall:.4f}")

    # Display F1-score
    print(f"F1-score : {f1:.4f}")

    # Display detailed classification report
    # It contains precision, recall, F1-score, and support
    print("\nClassification report:")

    print(classification_report(
        y_test, predictions, target_names=["Fake", "Real"], zero_division=0
    ))

    # Display confusion matrix
    #
    # Rows represent actual labels
    # Columns represent predicted labels
    #
    # It helps us understand correct and incorrect predictions
    print("Confusion matrix [rows=true, columns=predicted]:")

    print(confusion_matrix(y_test, predictions))

    # ========================================================
    # 11. SAVE THE TRAINED MODEL
    # ========================================================

    # Save the complete trained pipeline using joblib
    # compress=3 applies compression to the saved model file
    joblib.dump(model, MODEL_PATH, compress=3)

    # ========================================================
    # 12. SAVE MODEL EVALUATION METRICS
    # ========================================================

    # Create a dictionary containing the model evaluation results
    metrics = {

        # Store accuracy
        "accuracy": float(accuracy),

        # Store precision
        "precision": float(precision),

        # Store recall
        "recall": float(recall),

        # Store F1-score
        "f1_score": float(f1),

        # Store the number of testing articles
        "test_size": int(len(y_test)),

        # Store the total number of usable articles
        "total_articles": int(len(data)),

        # Explain the meaning of numerical labels
        "label_meaning": {"0": "Fake", "1": "Real"}
    }

    # Open the metrics.json file in write mode
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:

        # Convert the metrics dictionary into JSON format
        # indent=2 makes the JSON file easier to read
        json.dump(metrics, f, indent=2)

    # ========================================================
    # 13. DISPLAY FINAL MESSAGES
    # ========================================================

    # Display the location where the trained model was saved
    print(f"\nSaved model to: {MODEL_PATH}")

    # Inform the user that evaluation metrics were saved
    print("Saved evaluation metrics to: models/metrics.json")

    # Display the next step for running the graphical interface
    print("\nNext: run  streamlit run app.py")


# ============================================================
# 14. PROGRAM EXECUTION
# ============================================================

# This condition checks whether this file is being executed directly
#
# If it is executed directly, Python calls the main() function
# If it is imported into another Python file, main() will not run automatically

if __name__ == "__main__":

    # Start the model training process
    main()