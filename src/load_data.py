from pathlib import Path

import gdown


def download_google_drive_folder(folder_url: str, output_dir: str = "data"):
    """
    Download all files from a Google Drive folder.

    Args:
        folder_url (str): URL of the Google Drive folder
        output_dir (str): Directory to save the downloaded files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract folder ID from URL
    folder_id = folder_url.split("/")[-1]

    # Download all files from the folder
    gdown.download_folder(folder_url, output=output_dir, quiet=False, use_cookies=False)

    print(f"Download completed. Files saved to {output_dir}")


if __name__ == "__main__":
    # Google Drive folder URL
    folder_url = (
        "https://drive.google.com/drive/folders/1G6KdcUhyH15xV6AmcaDTb9i0Hxjtcvdp"
    )

    # Download the files
    download_google_drive_folder(folder_url)
