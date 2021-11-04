import subprocess
from threading import Thread
from typing import List


def run_command(command: str, thread: bool = False):
    """Default method for executing shell commands with pytest."""
    command = command.split(" ")
    if thread:
        t = Thread(target=subprocess.run, args=(command,))
        t.start()
        return t
    else:
        subprocess.run(command)
        return None
