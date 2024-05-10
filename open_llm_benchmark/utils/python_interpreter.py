
import subprocess
import sys


def python_interpreter(code: str, timeout: int=None):
    "Excute the python code and return the content printed during running process or error information."
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if not result.stderr:
            return {"success": True, "content": result.stdout.strip()}
        else:
            return {"success": False, "content": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"success": False, "content": f"TimeoutExpired: code time out after {timeout} seconds."}

