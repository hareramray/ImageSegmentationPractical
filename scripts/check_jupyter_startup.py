"""Start JupyterLab without a browser, verify it responds, then stop it."""
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
command = [sys.executable, "-m", "jupyterlab", "--no-browser", "--ServerApp.ip=127.0.0.1",
           f"--ServerApp.port={port}", "--ServerApp.port_retries=0",
           f"--ServerApp.root_dir={root}", "--LabApp.default_url=/lab/tree/notebooks/00_start_here.ipynb"]
process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
try:
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("JupyterLab exited before startup; inspect installation/configuration")
        try:
            with urlopen(f"http://127.0.0.1:{port}/lab", timeout=1) as response:
                assert response.status == 200
            print("JupyterLab startup and browser route: passed (temporary server stopped after check)")
            break
        except HTTPError as error:
            if error.code in (401, 403):
                print("JupyterLab startup: passed (authentication required, as expected)")
                break
            raise
        except (URLError, TimeoutError):
            time.sleep(0.3)
    else:
        raise TimeoutError("JupyterLab did not become ready within 25 seconds")
finally:
    if process.poll() is None:
        process.terminate()
    process.communicate(timeout=10)  # Do not print logs containing authentication URLs.
