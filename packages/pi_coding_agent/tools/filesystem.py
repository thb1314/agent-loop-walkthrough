"""基于 POSIX openat 语义的项目只读文件系统。"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

from packages.errors import MiniPiError


class ProjectFiles:
    """从可信根描述符出发，拒绝符号链接与目录逃逸。"""

    def __init__(self, root: Path, max_read_bytes: int) -> None:
        if (
            os.name != "posix"
            or not hasattr(os, "O_NOFOLLOW")
            or os.open not in os.supports_dir_fd
            or os.stat not in os.supports_dir_fd
            or os.stat not in os.supports_follow_symlinks
        ):
            raise MiniPiError("安全文件工具需要支持 openat/O_NOFOLLOW 的 POSIX 系统")

        self.root = Path(os.path.abspath(os.fspath(root.expanduser())))
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor: int | None = None
        try:
            descriptor = os.open("/", flags)
            for part in self.root.parts[1:]:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
        except (OSError, RuntimeError, ValueError) as error:
            if descriptor is not None:
                os.close(descriptor)
            raise MiniPiError(f"无法安全打开项目目录：{error}") from error

        self._root_fd = descriptor
        self._closed = False
        self.max_read_bytes = max_read_bytes

    def close(self) -> None:
        if not getattr(self, "_closed", True):
            os.close(self._root_fd)
            self._closed = True

    def __enter__(self) -> ProjectFiles:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _parts(raw_path: Any) -> list[str]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise MiniPiError("工具参数 path 必须是非空字符串")
        if "\x00" in raw_path:
            raise MiniPiError("工具参数 path 不能包含 NUL 字符")
        path = Path(raw_path)
        if path.is_absolute():
            raise MiniPiError(
                f"路径越界：{raw_path}；必须使用相对项目根目录的路径，根目录使用 .，禁止使用 /"
            )
        parts: list[str] = []
        for part in path.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise MiniPiError(
                    f"路径越界：{raw_path}；路径不能包含 ..，只能使用项目内相对路径"
                )
            parts.append(part)
        return parts

    def _open_directory(self, parts: list[str]) -> int:
        if self._closed:
            raise MiniPiError("文件工具已经关闭")
        descriptor = os.dup(self._root_fd)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            for part in parts:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except (OSError, RuntimeError, ValueError) as error:
            os.close(descriptor)
            raise MiniPiError(f"无法安全打开目录：{error}") from error

    def list_files(self, raw_path: Any) -> str:
        parts = self._parts(raw_path)
        descriptor = self._open_directory(parts)
        try:
            entries: list[tuple[int, str, str]] = []
            prefix = "/".join(parts)
            prefix = f"{prefix}/" if prefix else ""
            for name in os.listdir(descriptor):
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    kind, suffix = 0, "/"
                elif stat.S_ISLNK(info.st_mode):
                    kind, suffix = 2, "@"
                else:
                    kind, suffix = 1, ""
                entries.append((kind, name.casefold(), f"{prefix}{name}{suffix}"))
        except (OSError, RuntimeError) as error:
            raise MiniPiError(f"无法列出目录 {raw_path}：{error}") from error
        finally:
            os.close(descriptor)
        entries.sort()
        return "\n".join(entry[2] for entry in entries) if entries else "(空目录)"

    def read_file(self, raw_path: Any) -> str:
        parts = self._parts(raw_path)
        if not parts:
            raise MiniPiError(f"不是普通文件：{raw_path}")

        parent = self._open_directory(parts[:-1])
        # 先以非阻塞方式打开，避免在 fstat 拒绝命名管道前挂起。
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        try:
            descriptor = os.open(parts[-1], flags, dir_fd=parent)
        except (OSError, ValueError) as error:
            raise MiniPiError(f"无法安全打开文件 {raw_path}：{error}") from error
        finally:
            os.close(parent)

        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise MiniPiError(f"不是普通文件：{raw_path}")
            if info.st_size > self.max_read_bytes:
                raise MiniPiError(
                    f"文件过大：{raw_path} 为 {info.st_size} 字节，"
                    f"限制为 {self.max_read_bytes} 字节"
                )

            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(64 * 1024, self.max_read_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.max_read_bytes:
                    raise MiniPiError(
                        f"读取时文件超过限制：{raw_path}，限制为 {self.max_read_bytes} 字节"
                    )
            raw = b"".join(chunks)
        except OSError as error:
            raise MiniPiError(f"无法读取文件 {raw_path}：{error}") from error
        finally:
            os.close(descriptor)

        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MiniPiError(f"文件不是 UTF-8 文本：{raw_path}") from error
