# -*- coding: utf-8 -*-
"""
统一的机械臂任务占用、取消与状态记录 (Motion Task Runtime)

所有会驱动机械臂的入口 (点动、工位移动、抓取、回放、巡检、示教切换、象棋物理执行、
OpenVLA 步进) 都必须先通过 TaskManager 原子地取得占用，才能下发动作。

- try_start(): 原子取得占用并在后台线程中执行任务，忙碌时抛出 RobotBusyError；
- run_sync():  原子取得占用并在当前线程中执行短任务 (点动、夹爪等)；
- session():   上下文管理器形式，供已有工作线程 (如象棋状态机) 在物理执行阶段占用机械臂；
- cancel_current(): 设置取消信号，任务在工序切换与等待期间通过 CancelToken 检查并退出。

任务结束、取消和故障后的状态释放集中在 _finish() 中处理，调用方不再分别维护忙碌标志。
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional


class RobotBusyError(RuntimeError):
    """机械臂已被其他任务占用。"""


class TaskCancelled(RuntimeError):
    """任务已被取消 (停止请求或用户取消)。"""


# 任务状态
PENDING = "pending"
RUNNING = "running"
CANCELLING = "cancelling"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"

FINAL_STATES = (SUCCEEDED, FAILED, CANCELLED)


class CancelToken:
    """任务取消信号。工序切换、等待和逆解搜索时调用 check()/wait()。"""

    def __init__(self, external_events: Optional[List[threading.Event]] = None):
        self._event = threading.Event()
        self.reason: str = ""
        self._external = [e for e in (external_events or []) if e is not None]

    def cancel(self, reason: str = "任务已取消") -> None:
        if not self._event.is_set():
            self.reason = reason
        self._event.set()

    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        for ev in self._external:
            if ev.is_set():
                if not self.reason:
                    self.reason = "已请求急停"
                return True
        return False

    def check(self) -> None:
        """若已取消则抛出 TaskCancelled。"""
        if self.is_cancelled():
            raise TaskCancelled(self.reason or "任务已取消")

    def wait(self, seconds: float, step: float = 0.05) -> None:
        """可中断的等待；取消时立即抛出 TaskCancelled。"""
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            self.check()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._event.wait(min(step, remaining))

    # 兼容 threading.Event 风格
    def is_set(self) -> bool:
        return self.is_cancelled()


class NullToken(CancelToken):
    """不会被取消的占位令牌 (仍会响应外部急停事件)。"""


class TaskRecord:
    """单个任务的编号、类型、状态、进度与错误原因。"""

    def __init__(self, task_id: int, task_type: str, description: str = ""):
        self.id = task_id
        self.type = task_type
        self.description = description
        self.state = PENDING
        self.progress = 0
        self.step = ""
        self.error: Optional[str] = None
        self.result: Any = None
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.token = CancelToken()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "state": self.state,
            "progress": self.progress,
            "step": self.step,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "cancel_reason": self.token.reason or None,
        }


class TaskManager:
    """机械臂唯一占用者管理。线程安全。"""

    def __init__(self, stop_event_provider: Optional[Callable[[], Optional[threading.Event]]] = None,
                 on_change: Optional[Callable[[Optional[TaskRecord]], None]] = None,
                 history_size: int = 30):
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._current: Optional[TaskRecord] = None
        self._history: List[TaskRecord] = []
        self._history_size = history_size
        self._stop_event_provider = stop_event_provider
        self._on_change = on_change
        self._threads: Dict[int, threading.Thread] = {}
        self._accepting = True

    # ------------------------------------------------------------------ 查询
    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._current is not None

    def current(self) -> Optional[TaskRecord]:
        with self._lock:
            return self._current

    def current_dict(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._current.to_dict() if self._current else None

    def get(self, task_id: int) -> Optional[TaskRecord]:
        with self._lock:
            if self._current and self._current.id == task_id:
                return self._current
            for rec in reversed(self._history):
                if rec.id == task_id:
                    return rec
        return None

    def last(self) -> Optional[TaskRecord]:
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._history]

    # ------------------------------------------------------------------ 占用
    def _stop_event(self) -> Optional[threading.Event]:
        try:
            return self._stop_event_provider() if self._stop_event_provider else None
        except Exception:
            return None

    def _acquire(self, task_type: str, description: str) -> TaskRecord:
        with self._lock:
            if not self._accepting:
                raise RobotBusyError("控制服务正在关闭，不再接受新任务")
            stop_ev = self._stop_event()
            if stop_ev is not None and stop_ev.is_set():
                raise RobotBusyError("已请求急停，请检查机械臂并重新启动控制程序")
            if self._current is not None:
                cur = self._current
                raise RobotBusyError(
                    f"机械臂正在执行任务 #{cur.id}【{cur.type}】，请等待结束或使用急停")
            rec = TaskRecord(next(self._ids), task_type, description)
            rec.token = CancelToken([stop_ev] if stop_ev is not None else None)
            rec.state = RUNNING
            rec.started_at = time.time()
            self._current = rec
        self._notify(rec)
        return rec

    def _finish(self, rec: TaskRecord, state: str, error: Optional[str] = None, result: Any = None) -> None:
        with self._lock:
            # 取消请求优先: 被取消的任务即使中途抛出其他异常也记录为 cancelled
            if state != SUCCEEDED and rec.token.is_cancelled():
                state = CANCELLED
                error = error or rec.token.reason or "任务已取消"
            rec.state = state
            rec.error = error
            rec.result = result
            rec.ended_at = time.time()
            if state == SUCCEEDED:
                rec.progress = 100
            if self._current is rec:
                self._current = None
            self._history.append(rec)
            if len(self._history) > self._history_size:
                self._history.pop(0)
            self._threads.pop(rec.id, None)
        self._notify(rec)

    def _notify(self, rec: Optional[TaskRecord]) -> None:
        if self._on_change:
            try:
                self._on_change(rec)
            except Exception:
                pass

    def update(self, rec: TaskRecord, step: Optional[str] = None, progress: Optional[int] = None) -> None:
        with self._lock:
            if step is not None:
                rec.step = step
            if progress is not None:
                rec.progress = max(0, min(100, int(progress)))
        self._notify(rec)

    def try_start(self, task_type: str, fn: Callable[[CancelToken, TaskRecord], Any],
                  description: str = "", thread_factory: Callable[..., Any] = threading.Thread) -> TaskRecord:
        """原子取得占用后在新线程中执行 fn(token, record)。忙碌时抛出 RobotBusyError。"""
        rec = self._acquire(task_type, description)

        def runner():
            try:
                rec.token.check()
                result = fn(rec.token, rec)
                if isinstance(result, dict) and result.get("success") is False:
                    self._finish(rec, FAILED, str(result.get("error") or "任务未完成"), result)
                else:
                    self._finish(rec, SUCCEEDED, None, result)
            except TaskCancelled as e:
                self._finish(rec, CANCELLED, str(e))
            except Exception as e:  # noqa: BLE001 - 任务故障需完整记录
                traceback.print_exc()
                self._finish(rec, FAILED, str(e))

        try:
            t = thread_factory(target=runner, daemon=True)
            with self._lock:
                if isinstance(t, threading.Thread):
                    self._threads[rec.id] = t
            t.start()
        except Exception as e:
            self._finish(rec, FAILED, f"无法启动任务线程: {e}")
            raise
        return rec

    def run_sync(self, task_type: str, fn: Callable[[CancelToken, TaskRecord], Any], description: str = "") -> Any:
        """原子取得占用并在当前线程执行短任务，返回 fn 的结果；异常继续向上抛出。"""
        rec = self._acquire(task_type, description)
        try:
            result = fn(rec.token, rec)
        except TaskCancelled as e:
            self._finish(rec, CANCELLED, str(e))
            raise
        except Exception as e:
            self._finish(rec, FAILED, str(e))
            raise
        if isinstance(result, dict) and result.get("success") is False:
            self._finish(rec, FAILED, str(result.get("error") or "任务未完成"), result)
        else:
            self._finish(rec, SUCCEEDED, None, result)
        return result

    @contextmanager
    def session(self, task_type: str, description: str = ""):
        """上下文管理器: with tm.session("xiangqi_move") as token: ...  忙碌时抛出 RobotBusyError。"""
        rec = self._acquire(task_type, description)
        try:
            yield rec.token
        except TaskCancelled as e:
            self._finish(rec, CANCELLED, str(e))
            raise
        except Exception as e:
            self._finish(rec, FAILED, str(e))
            raise
        else:
            self._finish(rec, SUCCEEDED)

    # ------------------------------------------------------------------ 取消与关闭
    def cancel_current(self, reason: str = "任务已取消") -> Optional[TaskRecord]:
        with self._lock:
            rec = self._current
            if rec is None:
                return None
            rec.token.cancel(reason)
            if rec.state == RUNNING:
                rec.state = CANCELLING
        self._notify(rec)
        return rec

    def shutdown(self, timeout: float = 5.0, reason: str = "控制服务正在关闭") -> bool:
        """停止接受新任务，取消当前任务并有界等待工作线程退出。返回是否全部退出。"""
        with self._lock:
            self._accepting = False
            threads = list(self._threads.values())
        self.cancel_current(reason)
        deadline = time.monotonic() + timeout
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                t.join(remaining)
        return not any(t.is_alive() for t in threads)
