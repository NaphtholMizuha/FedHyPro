from contextlib import contextmanager
import time
import sys
import os

@contextmanager
def timer(name: str):
    """
    A timer context manager that measures the execution time of a block.
    It returns the elapsed time for external use.
    """
    start = time.time()  # Record the start time
    elapsed = None
    try:
        yield lambda: elapsed
    finally:
        end = time.time()  # Record the end time
        elapsed = end - start  # Calculate the elapsed time


@contextmanager
def suppress_output():
    # 保存当前的 stdout 和 stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        # 重定向 stdout 和 stderr 到 /dev/null
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        yield
    finally:
        # 恢复 stdout 和 stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
