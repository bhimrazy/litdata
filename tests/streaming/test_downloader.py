import contextlib
import os
from unittest import mock
from unittest.mock import MagicMock

from litdata.streaming.downloader import (
    AzureDownloader,
    GCPDownloader,
    HFDownloader,
    LocalDownloaderWithCache,
    S3Downloader,
    shutil,
    subprocess,
)


def test_s3_downloader_fast(tmpdir, monkeypatch):
    monkeypatch.setattr(os, "system", MagicMock(return_value=0))
    popen_mock = MagicMock()
    monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=popen_mock))
    downloader = S3Downloader(tmpdir, tmpdir, [])
    downloader.download_file("s3://random_bucket/a.txt", os.path.join(tmpdir, "a.txt"))
    popen_mock.wait.assert_called()


@mock.patch("litdata.streaming.downloader._GOOGLE_STORAGE_AVAILABLE", True)
def test_gcp_downloader(tmpdir, monkeypatch, google_mock):
    # Create mock objects
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.download_to_filename = MagicMock()

    # Patch the storage client to return the mock client
    google_mock.cloud.storage.Client = MagicMock(return_value=mock_client)

    # Configure the mock client to return the mock bucket and blob
    mock_client.bucket = MagicMock(return_value=mock_bucket)
    mock_bucket.blob = MagicMock(return_value=mock_blob)

    # Initialize the downloader
    storage_options = {"project": "DUMMY_PROJECT"}
    downloader = GCPDownloader("gs://random_bucket", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("gs://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    google_mock.cloud.storage.Client.assert_called_with(**storage_options)
    mock_client.bucket.assert_called_with("random_bucket")
    mock_bucket.blob.assert_called_with("a.txt")
    mock_blob.download_to_filename.assert_called_with(local_filepath)


@mock.patch("litdata.streaming.downloader._AZURE_STORAGE_AVAILABLE", True)
def test_azure_downloader(tmpdir, monkeypatch, azure_mock):
    mock_blob = MagicMock()
    mock_blob_data = MagicMock()
    mock_blob.download_blob.return_value = mock_blob_data
    service_mock = MagicMock()
    service_mock.get_blob_client.return_value = mock_blob

    azure_mock.storage.blob.BlobServiceClient = MagicMock(return_value=service_mock)

    # Initialize the downloader
    storage_options = {"project": "DUMMY_PROJECT"}
    downloader = AzureDownloader("azure://random_bucket", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")
    downloader.download_file("azure://random_bucket/a.txt", local_filepath)

    # Assert that the correct methods were called
    azure_mock.storage.blob.BlobServiceClient.assert_called_with(**storage_options)
    service_mock.get_blob_client.assert_called_with(container="random_bucket", blob="a.txt")
    mock_blob.download_blob.assert_called()
    mock_blob_data.readinto.assert_called()


@mock.patch("litdata.streaming.downloader._HUGGINGFACE_HUB_AVAILABLE", True)
def test_hf_downloader(tmpdir, monkeypatch, huggingface_mock):
    mock_hf_hub_download = MagicMock()
    huggingface_mock.hf_hub_download = mock_hf_hub_download

    # Initialize the downloader
    storage_options = {}
    downloader = HFDownloader("hf://datasets/random_org/random_repo", tmpdir, [], storage_options)
    local_filepath = os.path.join(tmpdir, "a.txt")

    # ignore filenotfound error for this test TODO: write a better test
    with contextlib.suppress(FileNotFoundError):
        downloader.download_file("hf://datasets/random_org/random_repo/a.txt", local_filepath)
    # Assert that the correct methods were called
    huggingface_mock.hf_hub_download.assert_called_once()
    huggingface_mock.hf_hub_download.assert_called_with(
        repo_id="random_org/random_repo", filename="a.txt", local_dir=tmpdir, repo_type="dataset"
    )

    # Test that the file is not downloaded if it already exists
    with contextlib.suppress(FileNotFoundError):
        downloader.download_file("hf://datasets/random_org/random_repo/a.txt", local_filepath)
        huggingface_mock.hf_hub_download.assert_not_called()


def test_download_with_cache(tmpdir, monkeypatch):
    # Create a file to download/cache
    with open("a.txt", "w") as f:
        f.write("hello")

    try:
        local_downloader = LocalDownloaderWithCache(tmpdir, tmpdir, [])
        shutil_mock = MagicMock()
        monkeypatch.setattr(shutil, "copy", shutil_mock)
        local_downloader.download_file("local:a.txt", os.path.join(tmpdir, "a.txt"))
        shutil_mock.assert_called()
    finally:
        os.remove("a.txt")
