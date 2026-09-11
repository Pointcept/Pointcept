"""
Tests for scripts/train.sh and scripts/test.sh.

The scripts are exercised in a temporary copy of the repository layout with a
stub interpreter that echoes its argv, so no GPU, torch or cluster is needed.

Run with: python -m unittest tests/test_launch_scripts.py
"""

import os
import shutil
import stat
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUB = '#!/bin/sh\necho "STUB_ARGV $*"\n'


class LaunchScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="pointcept-scripts-")
        for name in ("scripts", "tools", "pointcept"):
            os.makedirs(os.path.join(self.root, name))
        for name in ("train.sh", "test.sh"):
            shutil.copy(
                os.path.join(REPO_ROOT, "scripts", name),
                os.path.join(self.root, "scripts", name),
            )
        self.stub = os.path.join(self.root, "stub_python")
        with open(self.stub, "w") as f:
            f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IEXEC)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def run_script(self, name, *args, env=None):
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("SLURM_") and k not in ("DIST_URL", "MACHINE_RANK")
        }
        clean_env.update(env or {})
        cmd = [
            "sh",
            os.path.join(self.root, "scripts", name),
            "-p",
            self.stub,
            "-g",
            "1",
            "-d",
            "dataset",
            "-c",
            "config",
            "-n",
            "exp",
        ] + list(args)
        return subprocess.run(
            cmd, cwd=self.root, env=clean_env, capture_output=True, text=True
        )

    def launch_argv(self, result):
        lines = [
            line for line in result.stdout.splitlines() if line.startswith("STUB_ARGV ")
        ]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return lines[0]

    def check_single_machine_default(self, name):
        result = self.run_script(name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.launch_argv(result)
        self.assertIn("--num-machines 1", argv)
        self.assertIn("--machine-rank 0", argv)
        self.assertIn("--dist-url auto", argv)

    def check_multi_machine_flags(self, name):
        result = self.run_script(
            name, "-m", "2", "-u", "tcp://10.0.0.1:29500", "-k", "1"
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.launch_argv(result)
        self.assertIn("--num-machines 2", argv)
        self.assertIn("--machine-rank 1", argv)
        self.assertIn("--dist-url tcp://10.0.0.1:29500", argv)

    def check_multi_machine_env(self, name):
        env = {"DIST_URL": "tcp://10.0.0.2:29501", "MACHINE_RANK": "1"}
        result = self.run_script(name, "-m", "2", env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.launch_argv(result)
        self.assertIn("--machine-rank 1", argv)
        self.assertIn("--dist-url tcp://10.0.0.2:29501", argv)

    def check_slurm_node_id_rank(self, name):
        result = self.run_script(
            name, "-m", "2", "-u", "tcp://10.0.0.1:29500", env={"SLURM_NODEID": "3"}
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--machine-rank 3", self.launch_argv(result))

    def check_slurm_explicit_url_wins(self, name):
        env = {"SLURM_NODELIST": "node[1-2]", "SLURM_NODEID": "1"}
        result = self.run_script(name, "-m", "2", "-u", "tcp://10.0.0.1:29500", env=env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = self.launch_argv(result)
        self.assertIn("--machine-rank 1", argv)
        self.assertIn("--dist-url tcp://10.0.0.1:29500", argv)

    def check_multi_machine_without_url_fails(self, name):
        result = self.run_script(name, "-m", "2")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("-u", result.stderr)
        self.assertIn("DIST_URL", result.stderr)
        self.assertNotIn("STUB_ARGV", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.root, "exp")))

    def test_train_single_machine_default(self):
        self.check_single_machine_default("train.sh")

    def test_train_multi_machine_flags(self):
        self.check_multi_machine_flags("train.sh")

    def test_train_multi_machine_env(self):
        self.check_multi_machine_env("train.sh")

    def test_train_slurm_node_id_rank(self):
        self.check_slurm_node_id_rank("train.sh")

    def test_train_slurm_explicit_url_wins(self):
        self.check_slurm_explicit_url_wins("train.sh")

    def test_train_multi_machine_without_url_fails(self):
        self.check_multi_machine_without_url_fails("train.sh")

    def test_test_single_machine_default(self):
        self.check_single_machine_default("test.sh")

    def test_test_multi_machine_flags(self):
        self.check_multi_machine_flags("test.sh")

    def test_test_multi_machine_env(self):
        self.check_multi_machine_env("test.sh")

    def test_test_slurm_node_id_rank(self):
        self.check_slurm_node_id_rank("test.sh")

    def test_test_slurm_explicit_url_wins(self):
        self.check_slurm_explicit_url_wins("test.sh")

    def test_test_multi_machine_without_url_fails(self):
        self.check_multi_machine_without_url_fails("test.sh")


if __name__ == "__main__":
    unittest.main()
