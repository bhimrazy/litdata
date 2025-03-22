import time

from tqdm import tqdm


class ProgressMonitor:
    def __init__(self, file_name: str, total_bytes: int):
        self.file_name = file_name
        self.total_bytes = total_bytes
        self.progress_bar = None

    def __call__(self, bytes_transferred: int) -> None:
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.total_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {self.file_name}",
                dynamic_ncols=True,
                leave=True,
            )
        self.progress_bar.update(bytes_transferred)
        self.progress_bar.refresh()

    def close(self):
        if self.progress_bar is not None:
            self.progress_bar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
