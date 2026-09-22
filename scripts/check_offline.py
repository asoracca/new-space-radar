"""Offline CI payload. Invoke after the pinned environment has been installed."""
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
subprocess.run([sys.executable,'-m','pytest'],cwd=ROOT,check=True)
# Keep the standalone demo socket-blocked too, even outside Linux unshare.
import pytest_socket
pytest_socket.disable_socket()
from src.export import export
with tempfile.TemporaryDirectory() as temp:
    result=export(Path(temp)/'evidence',Path(temp)/'evidence.sqlite')
    assert result['counts']['synthetic']['included']==2
    assert result['counts']['historical']['included']==0
print('Offline tests, SQLite reconciliation and demo passed.')
