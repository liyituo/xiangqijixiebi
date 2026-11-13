# -*- coding: utf-8 -*-
"""
JSON 配置与数据文件的原子读写工具 (json_store)

- atomic_write_json(): 先写同目录临时文件并 fsync，再用 os.replace 整体替换，避免写入中断留下半个文件；
- update_json(): 在进程内锁 + 跨进程锁文件保护下读取-合并-写回，保留未知字段，处理多入口写入冲突；
- read_json(): 读取失败时返回 (None, 错误原因)，调用方据此报告"不可用"状态而不是静默使用默认值。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Tuple

_process_locks: Dict[str, threading.RLock] = {}
_process_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.RLock:
    key = os.path.abspath(path)
    with _process_locks_guard:
        lk = _process_locks.get(key)
        if lk is None:
            lk = threading.RLock()
            _process_locks[key] = lk
        return lk


@contextmanager
def file_lock(path: str, timeout: float = 5.0, stale_after: float = 30.0):
    """进程内 + 跨进程 (基于 O_EXCL 锁文件) 的互斥锁。"""
    plock = _lock_for(path)
    if not plock.acquire(timeout=timeout):
        raise TimeoutError(f"等待文件锁超时: {path}")
    lock_path = os.path.abspath(path) + ".lock"
    fd = None
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
                break
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(lock_path) > stale_after:
                        os.remove(lock_path)
                        continue
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"文件正被其他程序写入: {path}")
                time.sleep(0.05)
        yield
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.remove(lock_path)
            except OSError:
                pass
        plock.release()


def read_json(path: str) -> Tuple[Optional[Any], Optional[str]]:
    """读取 JSON；返回 (数据, None) 或 (None, 错误原因)。"""
    if not os.path.exists(path):
        return None, f"文件不存在: {os.path.basename(path)}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"文件内容损坏 (JSON 解析失败: {e.msg} 第 {e.lineno} 行)"
    except OSError as e:
        return None, f"文件读取失败: {e}"


def atomic_write_json(path: str, data: Any, indent: int = 2) -> None:
    """原子写入 JSON (临时文件 + fsync + os.replace)。失败时抛出异常，原文件保持不变。"""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def update_json(path: str, mutator: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
                default: Optional[Dict[str, Any]] = None, allow_corrupt_overwrite: bool = False) -> Dict[str, Any]:
    """
    加锁读取-修改-原子写回。mutator 接收当前字典 (可原地修改或返回新字典)。
    原文件损坏时默认抛出异常，避免把损坏配置覆盖成仅含部分字段的新文件。
    """
    with file_lock(path):
        data, err = read_json(path)
        if data is None:
            if os.path.exists(path) and not allow_corrupt_overwrite:
                raise ValueError(err or "配置文件不可读")
            data = dict(default or {})
        if not isinstance(data, dict):
            raise ValueError("配置文件顶层结构不是 JSON 对象")
        new_data = mutator(data)
        if new_data is None:
            new_data = data
        atomic_write_json(path, new_data)
        return new_data
