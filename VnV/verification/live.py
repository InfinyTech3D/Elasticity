"""Live figures in a process of their own, so they outlive the runner instead of dying with it.

A window owned by the runner leaves two bad options: block at the end and the terminal is held hostage,
or exit and the windows go with it. A second process settles it. matplotlib and the windows belong to
the child; the runner writes it one JSON line per level over a pipe and, when the sweep is over, closes
the pipe and walks away. The child keeps its windows interactive until they are closed, then exits.

The child is detached from the terminal's session, so closing the terminal or a Ctrl-C aimed at the
runner does not take the figures down with it. It imports nothing from SOFA -- only the drawing side of
this package -- so it starts in the time matplotlib takes to load.

Run as a module (`python -m VnV.verification.live`) it *is* the child; imported, it is the client.
"""

import json
import pathlib
import subprocess
import sys

# The plugin root, which is what has to be importable for the child's `-m` to resolve this package.
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _detached():
    """Popen arguments that put the child outside the terminal's session and signal handling."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    return {"start_new_session": True}      # no SIGHUP when the terminal closes, no shared Ctrl-C


class LiveWindows:
    """Client side: start the plot process, then hand it each level as its solve returns.

    One process serves the whole run, so `--all` ends with one window per deck rather than one process
    per deck. Sending is fire-and-forget: the child is not waited on anywhere, which is the point.
    """

    def __init__(self, log_path):
        # A detached child's traceback has nowhere to go -- no console on Windows, and stderr here
        # would land in the middle of a table -- so it goes to a file next to the figures.
        self.log = open(log_path, "w")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "VnV.verification.live"], cwd=str(PACKAGE_ROOT),
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log, text=True,
            **_detached())
        self.broken = False

    def figure(self, deck, levels, tolerance):
        """Open a window for one deck. `levels` up front keeps the colour of a level fixed for good."""
        self._send({"figure": deck, "levels": levels, "tolerance": tolerance})

    def add(self, deck, label, curves):
        self._send({"figure": deck, "label": label, "curves": curves})

    def detach(self):
        """Close the pipe and leave the child to it -- deliberately no wait(), so the shell comes back."""
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.log.close()

    def _send(self, message):
        if self.broken:
            return
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            # The windows were closed, or the child never came up. Neither is worth stopping a sweep
            # for; say it once and let the run finish.
            self.broken = True
            print("  the live plot process is gone -- see live.log; the run continues")


def _read(messages):
    """Feed the queue from the pipe, then a None to say the runner is done sending."""
    for line in sys.stdin:
        if line.strip():
            messages.put(json.loads(line))
    messages.put(None)


def serve():
    """Child side: draw what arrives, and hold the windows once the runner has gone.

    The pipe is read on a thread so the main thread stays free for the GUI: `plt.pause` in the idle
    branch is what keeps the windows responsive while a level is being solved on the other side of the
    pipe -- which the runner itself could never do, being busy inside SOFA.
    """
    import queue
    import threading
    import matplotlib.pyplot as plt
    from VnV.verification import plots

    # No display means no window to hold, and an Agg figure never closes, so the loop below would spin
    # forever on a figure nobody can see. Leave, and let the log say why.
    if not plots.interactive():
        print("no display: the plot process has nothing to show", file=sys.stderr)
        return

    messages = queue.Queue()
    threading.Thread(target=_read, args=(messages,), daemon=True).start()

    figures, streaming = {}, True
    # Stop only when the runner has finished sending *and* the last window has been closed.
    while streaming or plt.get_fignums():
        try:
            message = messages.get(timeout=0.05)
        except queue.Empty:
            plt.pause(0.05)
            continue

        if message is None:
            streaming = False
            continue

        deck = message["figure"]
        try:
            if "curves" in message:
                # JSON turned the Newton iteration keys into strings on the way over.
                curves = {int(key): values for key, values in message["curves"].items()}
                figures[deck].add(message["label"], curves)
            else:
                figures[deck] = plots.LivePlot(deck, message["levels"], message["tolerance"])
        except Exception as error:      # a closed window must not take the other decks down with it
            figures.pop(deck, None)
            print(f"dropped {deck}: {type(error).__name__}: {error}", file=sys.stderr)


if __name__ == "__main__":
    serve()
