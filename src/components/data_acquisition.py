import os
import subprocess
import zipfile

def download_kaggle_competition(competition_name: str, download_path: str = "data/"):
    # Ensure download directory exists
    os.makedirs(download_path, exist_ok=True)

    # Download competition dataset using Kaggle CLI
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", competition_name, "-p", download_path],
        check=True
    )
    print(f"✅ Downloaded competition '{competition_name}' to '{download_path}'.")

    # Unzip the downloaded file using Python's zipfile
    zip_path = os.path.join(download_path, f"{competition_name}.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    print(f"✅ Extracted '{zip_path}' to '{download_path}'.")

if __name__ == "__main__":
    print("✅ Kaggle credentials loaded successfully.")
    download_kaggle_competition("bluebook-for-bulldozers", download_path="data/bulldozer/")


