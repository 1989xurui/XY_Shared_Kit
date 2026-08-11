"""Sandbox: run an untrusted solution in a (soft-)isolated subprocess.

Hardening notes (post D7 audit):
- Child runs in its own process group so timeouts can kill the whole tree
  (grandchildren included) instead of only the direct child.
- On Windows we use CREATE_NEW_PROCESS_GROUP + `taskkill /F /T`;
  on POSIX we use start_new_session + os.killpg.
- PYTHONPATH is stripped from the child env so a candidate cannot
  `import hyra` / the host framework; only `import solution` (cwd) works.
- This is SOFT isolation for trusted-local use. True memory/CPU capping
  needs a container/job-object and is out of scope here.
"""
import os
import subprocess
import sys
import tempfile


def _clean_env():
    env = {}
    for k, v in os.environ.items():
        if k.upper() == "PYTHONPATH":
            continue  # prevent candidate importing the host framework
        env[k] = v
    return env


def _kill_tree(pid):
    if sys.platform == "win32":
        os.system(f"taskkill /F /T /PID {pid} >NUL 2>&1")
    else:
        try:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass


class Sandbox:
    def __init__(self, timeout=120):
        self.timeout = timeout

    def run(self, solution_code, runner_code, workdir=None):
        if workdir is None:
            workdir = tempfile.mkdtemp(prefix="hyra_sbox_")
        os.makedirs(workdir, exist_ok=True)
        sol_path = os.path.join(workdir, "solution.py")
        run_path = os.path.join(workdir, "runner.py")
        with open(sol_path, "w") as f:
            f.write(solution_code)
        with open(run_path, "w") as f:
            f.write(runner_code)

        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            proc = subprocess.Popen(
                [sys.executable, run_path],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_clean_env(),
                creationflags=creationflags,
                start_new_session=(sys.platform != "win32"),
            )
            try:
                out, err = proc.communicate(timeout=self.timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                _kill_tree(proc.pid)
                try:
                    out, err = proc.communicate(timeout=5)
                except Exception:
                    out, err = "", ""
                return {"ok": False, "stdout": "", "stderr": "TIMEOUT", "rc": -1}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "stdout": "", "stderr": repr(e), "rc": -2}

        return {
            "ok": rc == 0,
            "stdout": out.strip(),
            "stderr": err.strip(),
            "rc": rc,
        }
