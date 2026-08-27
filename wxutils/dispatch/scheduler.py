# -*- coding: utf-8 -*-


import threading
import uuid
import time

from typing import Optional, Dict, Any, List,TextIO

import os
import signal
from abc import ABC, abstractmethod
import weakref
from .job import pyro_expose

class BaseScheduler(ABC):
    """Core scheduler engine handling background threads, job queues, and process cleanup."""

    def __init__(self, max_concurrent_jobs: int = 2, poll_interval: float = 2.0):
        self.jobs: Dict[str, Any] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_concurrent_cores = 1
        self.poll_interval = poll_interval
        self.max_retries = 3

        self._running = True
        # self._monitor_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        # self._monitor_thread.start()
        self._stop_event = threading.Event()
        self._start_monitor_thread()
        # Finalizer handles both garbage collection (del) and process exit automatically.
        # No extra atexit registration needed as weakref.finalize handles atexit internally.
        self._finalizer = weakref.finalize(
            self,
            BaseScheduler._finalizer_routine,
            self.jobs,
            self._stop_event,
            self._monitor_thread
        )

    # -------------------------------------------------------------------------
    # Internal Loop & Queue Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def _start_job(self, job: Any) -> None:
        """Child classes MUST implement this method to handle process execution."""
        pass

    def _start_monitor_thread(self) -> None:
        """Starts monitoring thread without keeping a strong reference to self."""
        weak_self = weakref.ref(self)
        stop_event = self._stop_event

        def _thread_loop():
            while not stop_event.is_set():
                inst = weak_self()
                if inst is None or not inst._running:
                    break

                poll_interval = inst.poll_interval
                try:
                    inst._update_job_statuses()
                    inst._process_queue()
                except Exception as e:
                    print(f"Error in scheduler loop: {e}")

                # 1. CRITICAL: Drop local strong reference BEFORE thread sleeps/waits
                del inst

                # 2. Wait on event (unblocks instantly when finalizer sets stop_event)
                if stop_event.wait(timeout=poll_interval):
                    break

        self._monitor_thread = threading.Thread(target=_thread_loop, daemon=True)
        self._monitor_thread.start()

    def _poll_job(self, job: Any) -> Dict[str, Any]:
        """Polls an active process, updates status on completion, and cleans up handles."""
        if job.status == "RUNNING" and job._proc:
            retcode = job._proc.poll()
            if retcode is not None:
                job.exit_code = retcode
                job.mark_finished("COMPLETED" if retcode == 0 else "FAILED")
                self._cleanup_job_handles(job)
        return job.to_dict()

    def _update_job_statuses(self) -> None:
        """Updates status for all currently running jobs via _poll_job."""
        running_jobs = [j for j in self.jobs.values() if j.status == "RUNNING"]
        for job in running_jobs:
            try:
                self._poll_job(job)
            except Exception as e:
                print(f"Error polling status for active job '{job.job_id}': {e}")
                job.mark_finished("FAILED")
                job.exit_code = -1
                self._cleanup_job_handles(job)

    def _process_queue(self) -> None:
        """Launches queued jobs up to max slots with launch retry tracking."""
        running_jobs = [j for j in self.jobs.values() if j.status == "RUNNING"]
        queued_jobs = [j for j in self.jobs.values() if j.status == "QUEUED"]
        available_slots = self.max_concurrent_jobs - len(running_jobs)

        for job in queued_jobs[:available_slots]:
            try:
                self._start_job(job)
            except Exception as e:
                job.retry_count += 1
                self._cleanup_job_handles(job)

                if job.retry_count <= self.max_retries:
                    print(
                        f"Warning: Launch failed for job '{job.job_id}' "
                        f"({job.retry_count}/{self.max_retries}). Retrying... Error: {e}"
                    )
                else:
                    print(
                        f"Failed to launch job '{job.job_id}' after "
                        f"{self.max_retries} attempts: {e}"
                    )
                    job.mark_finished("FAILED")
                    job.exit_code = -1

    
    def _generate_job_id(self, prefix: str = "job") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    # -------------------------------------------------------------------------
    # Static Resource Cleanup Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _cleanup_job_handles(job: Any) -> None:
        """Flushes/closes open log file handles and clears process references."""
        log_file = getattr(job, "_log_file", None)
        if log_file and not log_file.closed:
            try:
                log_file.flush()
                log_file.close()
            except (OSError, ValueError):
                pass
        job._proc = None

    @staticmethod
    def _cleanup_job_proc(job: Any, force: bool = True, timeout: float = 3.0) -> None:
        """Recursively kills the entire process tree (mpiexec + child python processes)."""
        proc = getattr(job, "_proc", None)
        if not proc or proc.poll() is not None:
            BaseScheduler._cleanup_job_handles(job)
            return

        # Attempt process tree kill via psutil (handles mpiexec + spawned workers)
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)

            for child in children:
                if force:
                    child.kill()
                else:
                    child.terminate()

            if force:
                parent.kill()
            else:
                parent.terminate()

        except ImportError:
            # Fallback to OS process group kill if psutil isn't installed
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL if force else signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
        except Exception:
            pass
        finally:
            BaseScheduler._cleanup_job_handles(job)

    @staticmethod
    def _cleanup_all_procs(jobs: dict, force: bool = True) -> None:
        """Terminates processes for all active jobs in the dictionary."""
        for job in list(jobs.values()):
            BaseScheduler._cleanup_job_proc(job, force=force)

    @staticmethod
    def _finalizer_routine(jobs: dict, stop_event: threading.Event, thread: threading.Thread) -> None:
        """Static callback executed by weakref.finalize on del or process exit."""
        BaseScheduler._cleanup_all_procs(jobs, force=False)
        if stop_event:
            stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            
        


@pyro_expose
class Scheduler(BaseScheduler):
    """User-facing and RPC-exposed management interface."""

    # -------------------------------------------------------------------------
    # Process & Job Controls
    # -------------------------------------------------------------------------

    @pyro_expose
    def kill_job(self, job: str | Any, include_queued: bool = True, force: bool = False) -> Dict[str, Any]:
        """Kills a single job using _cleanup_job_proc and updates handles/status."""
        job_obj = self.jobs[job] if isinstance(job, str) else job

        if job_obj.status == "RUNNING":
            self._cleanup_job_proc(job_obj, force=force)
            job_obj.mark_finished("KILLED")
            self._cleanup_job_handles(job_obj)
        elif job_obj.status == "QUEUED" and include_queued:
            job_obj.mark_finished("KILLED")

        return job_obj.to_dict()

    @pyro_expose
    def kill_active_jobs(self, include_queued: bool = True, force: bool = False) -> List[Dict[str, Any]]:
        """Kills all running and queued jobs."""
        for job in list(self.jobs.values()):
            self.kill_job(job, include_queued=include_queued, force=force)
        return [j.to_dict() for j in self.jobs.values()]

    @pyro_expose
    def cleanup(self) -> None:
        """Full cleanup method for explicit or RPC invocations."""
        self.kill_active_jobs(force=True)
        self.stop_scheduler()
        if self._finalizer.alive:
            self._finalizer()

    # -------------------------------------------------------------------------
    # Daemon Lifecycle Controls
    # -------------------------------------------------------------------------

    @pyro_expose
    def status_scheduler(self) -> bool:
        return self._monitor_thread.is_alive()

    @pyro_expose
    def start_scheduler(self) -> bool:
        if self.status_scheduler():
            print("Scheduler is already started")
        else:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._monitor_thread.start()
        return self.status_scheduler()

    @pyro_expose
    def stop_scheduler(self, timeout: float = 5.0) -> bool:
        self._running = False
        start_time = time.time()

        while self.status_scheduler():
            if time.time() - start_time > timeout:
                break
            time.sleep(0.1)
        return self.status_scheduler()

    @pyro_expose
    def restart_scheduler(self) -> bool:
        self.stop_scheduler()
        self.start_scheduler()
        return self.status_scheduler()

    # -------------------------------------------------------------------------
    # Configuration Setters
    # -------------------------------------------------------------------------

    @pyro_expose
    def set_poll_interval(self, poll_interval: float) -> None:
        self.poll_interval = poll_interval

    @pyro_expose
    def set_max_concurrent_jobs(self, max_concurrent_jobs: int) -> None:
        self.max_concurrent_jobs = max_concurrent_jobs

    @pyro_expose
    def set_max_concurrent_cores(self, max_concurrent_cores: int) -> None:
        self.max_concurrent_cores = max_concurrent_cores