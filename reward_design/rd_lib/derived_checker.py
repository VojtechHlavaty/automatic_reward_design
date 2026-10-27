"""Pipeline side of the live check of derived quantities: a persistent IsaacLab worker (one per run).

The worker (:mod:`isaaclab_backend.derived_check_server`) runs on one GPU with a few environments. ``check``
sends the derived quantities of a stage YAML and returns (errors, stats); requests from parallel stage
generators are serialized. If the worker cannot start, the derived quantities are only checked statically.
"""

import json
import logging
import os
import queue
import subprocess
import sys
import threading

MARK = "@@CHECK@@ "


class DerivedChecker:
    def __init__(self, python, root_dir, task, gpu, log_file, init_checkpoint=None, start_timeout=900,
                 check_timeout=300):
        self.python = python
        self.root_dir = root_dir
        self.task = task
        self.gpu = gpu
        self.log_file = log_file
        self.init_checkpoint = init_checkpoint
        self.start_timeout = start_timeout
        self.check_timeout = check_timeout
        self.process = None
        self.interface = None
        self.audit = {}  # task facts for the trajectory audit of the check: success_flag, gamma
        self._answers = queue.Queue()
        self._lock = threading.Lock()
        self._next_id = 0

    # ---- worker process

    def start(self):
        cmd = [self.python, "-m", "isaaclab_backend.derived_check_server", "--task", self.task, "--headless"]
        if self.init_checkpoint:
            cmd += ["--init_checkpoint", self.init_checkpoint]
        env = dict(os.environ)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        self._log = open(self.log_file, "a")
        # a fresh answer queue per worker: the reader of a closed worker puts its exit marker (None) into its own
        # queue, not into the queue of the new worker (it made a restart look like a failed start)
        self._answers = queue.Queue()
        self.process = subprocess.Popen(
            cmd, cwd=self.root_dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        threading.Thread(target=self._reader, args=(self.process, self._answers, self._log), daemon=True).start()
        try:
            first = self._answers.get(timeout=self.start_timeout)
        except queue.Empty:
            first = None
        if not first or not first.get("ready"):
            logging.warning(f"The derived-quantity checker did not start (see {self.log_file}); "
                            "derived quantities are only checked statically")
            self.close()
            return False
        self.interface = first["interface"]
        logging.info(f"Derived-quantity checker ready on GPU {self.gpu} ({self.task})")
        return True

    @staticmethod
    def _reader(process, answers, log):
        for line in process.stdout:
            if line.startswith(MARK):
                try:
                    answers.put(json.loads(line[len(MARK):]))
                except json.JSONDecodeError:
                    pass
            else:
                log.write(line)
                log.flush()
        answers.put(None)  # the worker exited

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    # ---- checks

    def check(self, stage, adapter, progress_flags=None, term_table=False):
        """(errors, stats) of the stage on live environment states.

        Runs the stage's derived quantities (errors, NaNs, constant values, values outside the declared range) and,
        with ``progress_flags``, compares the stage reward between the task's progress levels: a later level must
        not earn clearly less per step than an earlier one.
        """
        derived = stage.get("derived") or []
        if not derived and not progress_flags:
            return [], {}
        with self._lock:
            if not self.alive() and not self.start():
                return [], {}
            self._next_id += 1
            request_id = self._next_id
            request = {"id": request_id, "adapter": adapter, "derived": derived,
                       "stage": {k: v for k, v in stage.items() if k in ("reward", "derived")},
                       "progress_flags": list(progress_flags or []), "audit": self.audit}
            if term_table:
                request["term_table"] = True
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            while True:
                try:
                    answer = self._answers.get(timeout=self.check_timeout)
                except queue.Empty:
                    answer = None
                if answer is None:
                    self.close()
                    return ["the live check of the derived quantities did not finish (the code may be too slow or "
                            "crash the simulator); simplify it"], {}
                if answer.get("id") == request_id:
                    return answer.get("errors", []), answer.get("stats", {})

    def profile(self, adapter, progress_flags):
        """{quantity: {progress level: [p5, median, p95, samples]}} of the registered scalar quantities on states of
        the worker's policy, or {} if the worker is not available."""
        with self._lock:
            if not self.alive() and not self.start():
                return {}
            self._next_id += 1
            request_id = self._next_id
            request = {"id": request_id, "profile": True, "adapter": adapter,
                       "progress_flags": list(progress_flags or [])}
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            while True:
                try:
                    answer = self._answers.get(timeout=self.check_timeout)
                except queue.Empty:
                    answer = None
                if answer is None:
                    self.close()
                    return {}
                if answer.get("id") == request_id:
                    return answer.get("profile", {})

    def interface_text(self):
        """Environment interface for the stage prompt."""
        if not self.interface:
            return "(not available)"
        lines = []
        for name, asset in self.interface["assets"].items():
            if asset["type"] == "articulation":
                lines.append(f"- env.{name}: articulation; bodies: {', '.join(asset['bodies'])}")
                lines.append(f"  joints: {', '.join(asset['joints'])}")
            else:
                lines.append(f"- env.{name}: rigid object (use h.root_pos(env, '{name}') / h.root_quat(env, '{name}'))")
        lines.append("Per-environment tensors of env (read only):")
        lines += [f"- env.{name}: {desc}" for name, desc in sorted(self.interface["tensors"].items())]
        return "\n".join(lines)


def make_checker(cfg, root_dir, task, init_checkpoint, log_file):
    python = cfg.isaaclab_python or sys.executable
    gpus = list(cfg.gpus) if cfg.gpus else [None]
    return DerivedChecker(python, root_dir, task, gpus[0], log_file, init_checkpoint)
