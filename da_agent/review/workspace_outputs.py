"""Workspace output tracking, quarantine, and write-protection helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import uuid
from typing import Any, Dict, List, Optional


logger = logging.getLogger("da_agent.workspace_outputs")


class WorkspaceOutputManager:
    """Manage files created by accepted or risky execution steps.

    The manager is intentionally stateless. The agent owns task containers,
    trajectory, verified-output sets, and task loggers; this class only performs
    the file-system operations using those existing objects.
    """

    def compute_outputs_from_diff(self, mnt_dir: str, file_snapshot: List[str]) -> List[str]:
        """Return workspace-relative files that are new since file_snapshot."""
        current_files = self.snapshot_mnt_files(mnt_dir)
        snapshot_set = set(file_snapshot)
        return [f for f in current_files if f not in snapshot_set]

    @staticmethod
    def safe_workspace_rel_path(rel_path: str) -> str:
        """Normalize a workspace-relative path and reject path traversal."""
        if not rel_path:
            return ""
        norm = os.path.normpath(str(rel_path).replace("\\", "/"))
        if norm in ("", ".") or os.path.isabs(norm) or norm.startswith(".."):
            return ""
        return norm

    @staticmethod
    def is_hidden_or_internal_path(rel_path: str) -> bool:
        parts = [part for part in rel_path.replace("\\", "/").split("/") if part]
        return any(part.startswith(".") for part in parts)

    def new_regular_outputs(self, mnt_dir: str, file_snapshot: List[str]) -> List[str]:
        outputs = []
        for rel in self.compute_outputs_from_diff(mnt_dir, file_snapshot):
            norm = self.safe_workspace_rel_path(rel)
            if not norm or self.is_hidden_or_internal_path(norm):
                continue
            full = os.path.join(mnt_dir, norm)
            if os.path.isfile(full):
                outputs.append(norm)
        return outputs

    @staticmethod
    def safe_step_dir_name(step_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", step_id or "unknown_step")
        return safe[:120] or "unknown_step"

    def quarantine_step_outputs(
        self,
        *,
        task_id: str,
        step_id: str,
        file_snapshot: Optional[List[str]],
        task_envs: Dict[str, Any],
        task_quarantined_outputs: Dict[str, List[Dict[str, Any]]],
        task_loggers: Dict[str, logging.Logger],
    ) -> Dict[str, Any]:
        """Move files newly created by a risky step into a hidden untrusted area."""
        if file_snapshot is None:
            return {}
        env = task_envs.get(task_id)
        if env is None:
            return {}
        mnt_dir = env.mnt_dir
        new_files = self.new_regular_outputs(mnt_dir, file_snapshot)
        if not new_files:
            return {
                "applied": False,
                "step_id": step_id,
                "files": [],
                "message": "no newly created regular files to quarantine",
            }

        quarantine_root_rel = os.path.join(".acid_untrusted", self.safe_step_dir_name(step_id))
        quarantine_root = os.path.join(mnt_dir, quarantine_root_rel)
        os.makedirs(quarantine_root, exist_ok=True)

        mnt_real = os.path.realpath(mnt_dir)
        moved = []
        failed = []
        for rel in new_files:
            src = os.path.join(mnt_dir, rel)
            src_real = os.path.realpath(src)
            if not src_real.startswith(mnt_real + os.sep):
                failed.append({"original": rel, "error": "path escaped workspace"})
                continue
            if not os.path.isfile(src):
                continue
            dest = os.path.join(quarantine_root, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                base, ext = os.path.splitext(dest)
                dest = f"{base}.{uuid.uuid4().hex[:8]}{ext}"
            try:
                shutil.move(src, dest)
                moved.append({
                    "original": rel,
                    "quarantined": os.path.relpath(dest, mnt_dir),
                })
            except Exception as exc:
                failed.append({"original": rel, "error": str(exc)})

        info = {
            "applied": bool(moved),
            "step_id": step_id,
            "quarantine_dir": quarantine_root_rel,
            "files": moved,
            "failed": failed,
            "message": (
                "new files from risky step were moved out of the active workspace"
                if moved else "no files were moved"
            ),
        }
        task_quarantined_outputs.setdefault(task_id, []).append(info)
        if task_id in task_loggers:
            task_loggers[task_id].info(
                "[OUTPUT-QUARANTINE] " + json.dumps(info, ensure_ascii=False)
            )
        logger.info("[OUTPUT-QUARANTINE] " + json.dumps(info, ensure_ascii=False))
        return info

    def cleanup_exploration_outputs(
        self,
        *,
        task_id: str,
        step_id: str,
        file_snapshot: Optional[List[str]],
        task_envs: Dict[str, Any],
        task_loggers: Dict[str, logging.Logger],
    ) -> Dict[str, Any]:
        """Delete regular files newly created by an exploration action."""
        if file_snapshot is None:
            return {}
        env = task_envs.get(task_id)
        if env is None:
            return {}

        mnt_dir = env.mnt_dir
        new_files = self.new_regular_outputs(mnt_dir, file_snapshot)
        if not new_files:
            return {
                "applied": False,
                "step_id": step_id,
                "files": [],
                "message": "no newly created exploration files to clean",
            }

        mnt_real = os.path.realpath(mnt_dir)
        removed = []
        failed = []
        for rel in new_files:
            src = os.path.join(mnt_dir, rel)
            src_real = os.path.realpath(src)
            if not src_real.startswith(mnt_real + os.sep):
                failed.append({"path": rel, "error": "path escaped workspace"})
                continue
            if not os.path.isfile(src):
                continue
            try:
                os.remove(src)
                removed.append(rel)
            except Exception as exc:
                failed.append({"path": rel, "error": str(exc)})

        info = {
            "applied": bool(removed),
            "step_id": step_id,
            "files": removed,
            "failed": failed,
            "message": (
                "new files from exploration were removed from the active workspace"
                if removed else "no exploration files were removed"
            ),
        }
        if task_id in task_loggers:
            task_loggers[task_id].info(
                "[EXPLORATION-READONLY-CLEANUP] " + json.dumps(info, ensure_ascii=False)
            )
        logger.info("[EXPLORATION-READONLY-CLEANUP] " + json.dumps(info, ensure_ascii=False))
        return info

    @staticmethod
    def format_quarantine_prompt(quarantine_info: Dict[str, Any]) -> str:
        if not quarantine_info or not quarantine_info.get("applied"):
            return ""
        lines = [
            "# QUARANTINED OUTPUTS #",
            "Files newly produced by the risky source step were moved out of the active workspace before this review.",
            "Treat these files as untrusted. Recompute from original/source data or explicitly verify before relying on their contents.",
        ]
        for item in quarantine_info.get("files", [])[:12]:
            lines.append(f"- {item.get('original')} -> {item.get('quarantined')}")
        extra = len(quarantine_info.get("files", [])) - 12
        if extra > 0:
            lines.append(f"- ... {extra} more file(s) quarantined")
        return "\n".join(lines) + "\n"

    @staticmethod
    def annotate_step_quarantine(
        trajectory: List[Dict[str, Any]],
        step_id: str,
        quarantine_info: Dict[str, Any],
    ) -> None:
        if not quarantine_info:
            return
        for entry in reversed(trajectory):
            if entry.get("step_id") == step_id:
                entry["output_quarantine"] = quarantine_info
                break

    def mark_verified_outputs(
        self,
        *,
        task_id: str,
        step_id: str,
        file_snapshot: Optional[List[str]],
        task_envs: Dict[str, Any],
        task_verified_outputs: Dict[str, set],
        task_loggers: Dict[str, logging.Logger],
    ) -> List[str]:
        """Record accepted step outputs as verified and make them read-only."""
        if file_snapshot is None:
            return []
        env = task_envs.get(task_id)
        if env is None:
            return []
        outputs = self.new_regular_outputs(env.mnt_dir, file_snapshot)
        if not outputs:
            return []
        bucket = task_verified_outputs.setdefault(task_id, set())
        bucket.update(outputs)
        self.freeze_verified_outputs(
            task_id=task_id,
            task_envs=task_envs,
            task_verified_outputs=task_verified_outputs,
            rel_paths=outputs,
        )
        info = {"step_id": step_id, "verified_outputs": outputs}
        if task_id in task_loggers:
            task_loggers[task_id].info(
                "[VERIFIED-OUTPUTS] " + json.dumps(info, ensure_ascii=False)
            )
        logger.info("[VERIFIED-OUTPUTS] " + json.dumps(info, ensure_ascii=False))
        return outputs

    def freeze_verified_outputs(
        self,
        *,
        task_id: str,
        task_envs: Dict[str, Any],
        task_verified_outputs: Dict[str, set],
        rel_paths: Optional[List[str]] = None,
    ) -> None:
        env = task_envs.get(task_id)
        if env is None or getattr(env, "container", None) is None:
            return
        paths = rel_paths if rel_paths is not None else sorted(task_verified_outputs.get(task_id, set()))
        safe_paths = []
        for rel in paths or []:
            norm = self.safe_workspace_rel_path(rel)
            if not norm or self.is_hidden_or_internal_path(norm):
                continue
            if not os.path.exists(os.path.join(env.mnt_dir, norm)):
                continue
            container_path = env.work_dir.rstrip("/") + "/" + norm.replace(os.sep, "/")
            safe_paths.append(shlex.quote(container_path))
        if not safe_paths:
            return
        for start in range(0, len(safe_paths), 100):
            chunk = safe_paths[start:start + 100]
            cmd = "chmod a-w -- " + " ".join(chunk)
            try:
                exit_code, output = env.container.exec_run(
                    ["bash", "-c", cmd],
                    workdir=env.work_dir,
                    user="root",
                )
                if exit_code != 0:
                    logger.debug(f"Freeze verified outputs failed: {output.decode(errors='ignore').strip()}")
            except Exception as exc:
                logger.debug(f"Freeze verified outputs error: {exc}")

    @staticmethod
    def snapshot_mnt_files(mnt_dir: str) -> List[str]:
        """Snapshot relative file paths in mnt_dir before step execution."""
        files = []
        for root, dirs, filenames in os.walk(mnt_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in filenames:
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, mnt_dir)
                files.append(rel)
        return files

    @staticmethod
    def set_write_protection(
        mnt_dir: str,
        readonly: bool = True,
        container: Any = None,
        work_dir: str = "/workspace",
    ) -> None:
        """Set or remove write protection on existing files in mnt_dir."""
        if container is None:
            for root, dirs, files in os.walk(mnt_dir):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        if readonly:
                            mode = os.stat(fpath).st_mode
                            os.chmod(fpath, mode & ~0o222)
                        else:
                            mode = os.stat(fpath).st_mode
                            new_mode = mode | 0o200
                            if fname.endswith('.py'):
                                new_mode |= 0o111
                            os.chmod(fpath, new_mode)
                    except OSError as exc:
                        logger.debug(f"chmod failed for {fpath}: {exc}")
            return

        if readonly:
            cmd = f'find {work_dir} -not -path "*/\\.*" -type f -exec chmod a-w {{}} \\;'
        else:
            cmd = (
                f'find {work_dir} -not -path "*/\\.*" -type f -exec chmod u+w {{}} \\; && '
                f'find {work_dir} -not -path "*/\\.*" -type f -name "*.py" -exec chmod a+x {{}} \\;'
            )
        try:
            exit_code, output = container.exec_run(
                ["bash", "-c", cmd], workdir=work_dir, user="root"
            )
            if exit_code != 0:
                logger.debug(f"Container chmod failed: {output.decode(errors='ignore').strip()}")
        except Exception as exc:
            logger.debug(f"Container chmod error: {exc}")
