from pathlib import Path
import re
import string
import logging

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"


# ---------------- Logging ---------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- NLTK Setup ---------------- #

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------- Text Preprocessing ---------------- #

def preprocess_text(text):
    """Clean and normalize text."""

    try:
        text = str(text)

        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove punctuation
        text = text.translate(
            str.maketrans("", "", string.punctuation)
        )

        # Convert to lowercase
        text = text.lower()

        # Remove numbers
        text = re.sub(r"\b\d+\b", "", text)

        # Remove stopwords
        words = [
            word for word in text.split()
            if word not in STOP_WORDS
        ]

        # Lemmatization
        words = [
            LEMMATIZER.lemmatize(word)
            for word in words
        ]

        return " ".join(words)

    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        return text


def normalize_text(df):
    """Apply preprocessing to the content column."""

    try:
        if "content" not in df.columns:
            raise ValueError("Column 'content' not found")

        df = df.copy()

        df["content"] = (
            df["content"]
            .fillna("")
            .apply(preprocess_text)
        )

        return df

    except Exception as e:
        logger.exception(f"Error normalizing text: {e}")
        raise


# ---------------- Main Pipeline ---------------- #

def main():
    try:
        logger.info("Starting data preprocessing")

        # Load raw data
        train_data = pd.read_csv(
            RAW_DATA_DIR / "train.csv"
        )

        test_data = pd.read_csv(
            RAW_DATA_DIR / "test.csv"
        )

        logger.info("Raw data loaded successfully")

        # Preprocess
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)

        # Create interim directory
        INTERIM_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save processed data
        train_processed.to_csv(
            INTERIM_DATA_DIR / "train_processed.csv",
            index=False
        )

        test_processed.to_csv(
            INTERIM_DATA_DIR / "test_processed.csv",
            index=False
        )

        logger.info("Processed data saved successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

    except Exception as e:
        logger.exception(f"Data preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()