from pathlib import Path
import yaml
import pandas as pd
import logging

from sklearn.model_selection import train_test_split


# ---------------- Paths ---------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARAMS_FILE = PROJECT_ROOT / "params.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------- Logging ---------------- #

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(
    PROJECT_ROOT / "errors.log"
)
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ---------------- Load Parameters ---------------- #

def load_params(params_path):
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)

        logger.debug(
            f"Parameters loaded successfully from {params_path}"
        )

        return params["data_ingestion"]["test_size"]

    except FileNotFoundError:
        logger.error("File not found")
        raise

    except yaml.YAMLError:
        logger.error("YAML error")
        raise

    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        raise


# ---------------- Read Data ---------------- #

def read_data(url):
    try:
        return pd.read_csv(url)

    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        raise RuntimeError(
            f"Failed to read data: {e}"
        ) from e


# ---------------- Process Data ---------------- #

def process_data(df):
    try:
        df = df.drop(columns=["tweet_id"])

        df = df[
            df["sentiment"].isin(["happiness", "sadness"])
        ].copy()

        df["sentiment"] = df["sentiment"].replace({
            "happiness": 1,
            "sadness": 0
        })

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed to process data: {e}"
        ) from e


# ---------------- Save Data ---------------- #

def save_data(data_path, train_data, test_data):
    try:
        data_path.mkdir(
            parents=True,
            exist_ok=True
        )

        train_data.to_csv(
            data_path / "train.csv",
            index=False
        )

        test_data.to_csv(
            data_path / "test.csv",
            index=False
        )

        logger.info(
            f"Data saved successfully to {data_path}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to save data: {e}"
        ) from e


# ---------------- Main Pipeline ---------------- #

def main():
    try:
        # Load parameters
        test_size = load_params(PARAMS_FILE)

        # Load data
        url = (
            "https://raw.githubusercontent.com/"
            "campusx-official/jupyter-masterclass/"
            "refs/heads/main/tweet_emotions.csv"
        )

        df = read_data(url)

        logger.info("Data loaded successfully")

        # Process data
        final_df = process_data(df)

        logger.info("Data processed successfully")

        # Train-test split
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=42
        )

        # Save data
        save_data(
            RAW_DATA_DIR,
            train_data,
            test_data
        )

        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.error(
            f"Data ingestion failed: {e}"
        )
        raise


if __name__ == "__main__":
    main()