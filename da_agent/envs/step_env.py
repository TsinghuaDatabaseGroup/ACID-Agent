# Single-step Docker execution environment

import io
import logging
import os
import tarfile
import time
from typing import List, Optional, Set, Tuple

import docker
from docker.errors import ImageNotFound
from docker.models.containers import Container

from da_agent.agent.action import Action, Bash, Python, SQL, Terminate
from da_agent.controllers.python import PythonController
from da_agent.envs.utils import create_folder_if_not_exists

logger = logging.getLogger("da_agent.step_env")

START_UP_DELAY = 2
DEFAULT_TIME_OUT = 60
MAX_OBS_LENGTH = 6000
DEFAULT_WORK_DIR = "/workspace"
MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1GB — skip files larger than this


class StepEnv:
    """
    Single-step Docker execution environment.

    Each step gets its own container. Files are passed via docker cp
    (no volume mount). Container is completely isolated — files are
    copied in before execution and only changed files are copied out after.

    Lifecycle:
        1. start()       — create container (no mount)
        2. copy_in(dir)  — copy host files into container's work_dir
        3. snapshot()    — record baseline file set (before execution)
        4. execute(action) — run code in container
        5. copy_out(dir) — copy only changed/new files back to host
        6. stop()        — destroy container
    """

    def __init__(
        self,
        image_name: str = "da_agent-image",
        work_dir: str = DEFAULT_WORK_DIR,
    ):
        self.image_name = image_name
        self.work_dir = work_dir
        self.container: Optional[Container] = None
        self.controller: Optional[PythonController] = None
        self._baseline_files: Set[str] = set()
        self.last_execution_meta = {}

    def start(self) -> None:
        """Create and start a new container (no volume mount)."""
        client = docker.from_env()
        container_name = f"step_{time.monotonic_ns()}"

        extra_params = {
            "detach": True,
            "tty": True,
            "stdout": True,
            "stderr": True,
            "stdin_open": True,
        }

        try:
            image = client.images.get(self.image_name)
            self.container = client.containers.run(
                image=image, name=container_name, **extra_params
            )
        except ImageNotFound:
            logger.info(f"Image {self.image_name} not found, trying to pull...")
            image = client.images.pull(self.image_name)[0]
            self.container = client.containers.run(
                image=image, name=container_name, **extra_params
            )

        time.sleep(START_UP_DELAY)

        # Ensure work_dir exists in container
        self.container.exec_run(f"mkdir -p {self.work_dir}")

        self.controller = PythonController(
            container=self.container, work_dir=self.work_dir, mnt_dir=None
        )
        logger.info(f"StepEnv container started: {self.container.name}")

    def copy_in(self, src_dir: str) -> None:
        """
        Copy files from host directory into container's work_dir.

        Skips files larger than MAX_FILE_SIZE.

        Args:
            src_dir: Host directory whose contents will be copied into
                     the container's work_dir.
        """
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")

        if not os.path.isdir(src_dir):
            logger.warning(f"Source directory {src_dir} does not exist, skipping copy_in")
            return

        self._tar_into_container(src_dir, self.work_dir)
        logger.info(f"Copied files from {src_dir} into container {self.work_dir}")

    def snapshot(self) -> None:
        """
        Record baseline file set in container's work_dir.

        Call after copy_in, before execute. This enables copy_out to
        only return changed/new files instead of everything.
        """
        self._baseline_files = self._list_container_files()

    def execute(self, action: Action) -> Tuple[str, bool]:
        """
        Execute an action in the container.

        Returns:
            Tuple of (observation, done)
        """
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")

        try:
            done = False
            if isinstance(action, Bash):
                observation = self._execute_bash(action)
            elif isinstance(action, SQL):
                observation = self._execute_sql(action)
            elif isinstance(action, Python):
                observation = self._execute_python(action)
            elif isinstance(action, Terminate):
                observation = "Terminate"
                self.last_execution_meta = {
                    "exit_code": 0,
                    "stdout": observation,
                    "stderr": "",
                    "combined_output": observation,
                    "action_type": "Terminate",
                }
                done = True
            else:
                raise ValueError(f"Unrecognized action type {action.action_type}!")
            if not isinstance(action, Terminate):
                self.last_execution_meta = dict(getattr(self.controller, "last_execution_meta", {}) or {})
                self.last_execution_meta["action_type"] = type(action).__name__
        except TimeoutError as e:
            observation = str(e)
            self.last_execution_meta = {
                "exit_code": None,
                "stdout": "",
                "stderr": observation,
                "combined_output": observation,
                "timed_out": True,
            }

        observation = self._handle_observation(observation)
        return observation, done

    def copy_out(self, dest_dir: str, changed_only: bool = True,
                 must_save_paths: Optional[set] = None) -> List[str]:
        """
        Copy files from container's work_dir to host directory.

        Args:
            dest_dir: Host directory to copy files into.
            changed_only: If True, only copy files that were added or
                          modified since snapshot(). If no snapshot was
                          taken, copies everything.
            must_save_paths: Set of relative paths that must be saved
                             regardless of size (e.g., step outputs).
                             Other large files (>MAX_FILE_SIZE) are skipped.

        Returns:
            List of copied file relative paths.
        """
        if self.container is None:
            raise RuntimeError("Container not started. Call start() first.")

        create_folder_if_not_exists(dest_dir)
        self._must_save_paths = must_save_paths or set()

        if changed_only and self._baseline_files:
            current_files = self._list_container_files()
            changed = current_files - self._baseline_files
            if not changed:
                logger.info("No changed files to copy out")
                return []
            return self._copy_changed_files(changed, dest_dir)
        else:
            return self._copy_all_files(dest_dir)

    def stop(self) -> None:
        """Stop and remove the container."""
        if self.container is not None:
            try:
                self.container.stop()
                self.container.remove()
                logger.info("StepEnv container stopped and removed")
            except Exception as e:
                logger.warning(f"Failed to stop container: {e}")
            self.container = None
            self.controller = None
            self._baseline_files = set()

    # ---- File operations ----

    def _list_container_files(self) -> Set[str]:
        """List all files in container's work_dir recursively."""
        if self.container is None:
            return set()

        exit_code, output = self.container.exec_run(
            f"find {self.work_dir} -type f"
        )
        if exit_code != 0:
            return set()

        files = set()
        for line in output.decode("utf-8", errors="ignore").strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip work_dir prefix
            if line.startswith(self.work_dir + "/"):
                rel = line[len(self.work_dir) + 1:]
            elif line.startswith(self.work_dir):
                rel = line[len(self.work_dir):].lstrip("/")
            else:
                rel = line
            if rel:
                files.add(rel)
        return files

    def _copy_changed_files(self, changed_files: Set[str], dest_dir: str) -> List[str]:
        """Copy specific changed files from container to host.

        Uses atomic write (write to temp then rename) so existing hard links
        in evidence directories are not corrupted when files are overwritten.
        """
        copied = []
        for rel_path in sorted(changed_files):
            container_path = f"{self.work_dir}/{rel_path}"
            try:
                stream, _ = self.container.get_archive(container_path)
                tar_stream = io.BytesIO()
                for chunk in stream:
                    tar_stream.write(chunk)
                tar_stream.seek(0)

                with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        # Skip large intermediate files (output files always saved)
                        if member.size > MAX_FILE_SIZE and rel_path not in self._must_save_paths:
                            logger.info(f"Skipping large file {rel_path} ({member.size} bytes)")
                            copied.append(rel_path)
                            break
                        dest_path = os.path.join(dest_dir, rel_path)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        src_file = tar.extractfile(member)
                        if src_file:
                            # Atomic write: temp file + rename preserves existing hard links
                            tmp_path = dest_path + ".tmp"
                            with open(tmp_path, "wb") as dst:
                                dst.write(src_file.read())
                            os.replace(tmp_path, dest_path)
                        copied.append(rel_path)
                        break  # Only need the file itself
            except Exception as e:
                logger.warning(f"Failed to copy {rel_path}: {e}")
        if copied:
            logger.info(f"Copied {len(copied)} changed files from container to {dest_dir}")
        return copied

    def _copy_all_files(self, dest_dir: str) -> List[str]:
        """Copy all files from container's work_dir to host.

        Uses atomic write to preserve existing hard links in evidence dirs.
        """
        try:
            stream, _ = self.container.get_archive(self.work_dir)
            tar_stream = io.BytesIO()
            for chunk in stream:
                tar_stream.write(chunk)
            tar_stream.seek(0)

            copied = []
            with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                for member in tar.getmembers():
                    name = member.name
                    # Strip top-level directory (e.g., "workspace/")
                    parts = name.split("/", 1)
                    if len(parts) > 1:
                        member.name = parts[1]
                    else:
                        continue
                    if not member.name:
                        continue
                    if not member.isfile():
                        continue
                    # Skip large intermediate files (output files always saved)
                    if member.size > MAX_FILE_SIZE and member.name not in self._must_save_paths:
                        logger.info(f"Skipping large file {member.name} ({member.size} bytes)")
                        continue

                    dest_path = os.path.join(dest_dir, member.name)
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    src_file = tar.extractfile(member)
                    if src_file:
                        tmp_path = dest_path + ".tmp"
                        with open(tmp_path, "wb") as dst:
                            dst.write(src_file.read())
                        os.replace(tmp_path, dest_path)
                        copied.append(member.name)

            logger.info(f"Copied {len(copied)} files from container to {dest_dir}")
            return copied
        except Exception as e:
            logger.error(f"Failed to copy files from container: {e}")
            return []

    def _tar_into_container(self, src_dir: str, dest_path: str) -> None:
        """Create tar from host dir and put into container. Skips large files."""
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            for item in os.listdir(src_dir):
                item_path = os.path.join(src_dir, item)
                # Skip large files
                if os.path.isfile(item_path) and os.path.getsize(item_path) > MAX_FILE_SIZE:
                    logger.info(f"Skipping large file {item} ({os.path.getsize(item_path)} bytes)")
                    continue
                tar.add(item_path, arcname=item)
        tar_stream.seek(0)
        self.container.put_archive(dest_path, tar_stream)

    # ---- Action execution ----

    def _execute_bash(self, action: Bash) -> str:
        obs = self.controller.execute_command(action.code)
        return obs if obs else "Command executed successfully. No output."

    def _execute_python(self, action: Python) -> str:
        obs = self.controller.execute_python_file(action.filepath, action.code)
        return obs if obs else f"{action.filepath} executed successfully. No output."

    def _execute_sql(self, action: SQL) -> str:
        obs = self.controller.execute_sql_code(action.file_path, action.code, action.output)
        return obs if obs else "SQL command executed successfully. No output."

    def _handle_observation(self, observation: str) -> str:
        if len(observation) > MAX_OBS_LENGTH:
            return observation[:MAX_OBS_LENGTH] + "\n[Observation too long, truncated.]"
        return observation
