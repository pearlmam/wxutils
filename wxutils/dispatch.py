# -*- coding: utf-8 -*-

# optional dependacy for enabling rpc use with the script
try:
    import Pyro5.api
    pyro_expose = Pyro5.api.expose
except ImportError:
    # No-op decorator if Pyro5 is not installed
    def pyro_expose(obj):
        return obj

import threading
import uuid
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List,TextIO
import subprocess
import os
import signal
import shutil
import dmanage.metadata.metastring as meta
from abc import ABC, abstractmethod
import inspect
import atexit
import weakref

defaultPyroDispatchHost = "localhost"
defaultPyroDispatchPort = 44444
defaultPyroDispatchName = "ProxyDispatch"

ONLY_EXPOSED = True
Pyro5.api.config.PICKLE_ENABLE = False


def check_path(path):
    path = Path(path)
    if path.is_dir():
        dir_path = path
        file_path=None
    elif path.is_file():
        dir_path = path.parent
        file_path=path.name
    else:
        dir_path = None
        file_path=None
    return dir_path, file_path

def load_job_config(inject_globals: bool = True, require_existing: bool = True) -> dict:
    config_raw = os.environ.get("JOB_CONFIG", "{}")
    config = json.loads(config_raw)

    is_managed = config.pop("_managed", False)

    # Return cleanly if run from terminal OR if scheduler passed no parameter overrides
    if not is_managed or not config:
        return {}

    if inject_globals:
        caller_globals = inspect.currentframe().f_back.f_globals
        modified_count = 0

        for key, new_val in config.items():
            if require_existing and key not in caller_globals:
                raise KeyError(
                    f"Parameter '{key}' in JOB_CONFIG is not defined in the input script. "
                    "Ensure default values are declared in the script before calling load_job_config()."
                )

            if caller_globals.get(key) != new_val:
                caller_globals[key] = new_val
                modified_count += 1

        if modified_count == 0:
            pass
            # raise ValueError(
            #     "JOB_CONFIG parameters were passed, but 0 global variables were modified "
            #     "(the overrides passed were identical to the script's hardcoded defaults)."
            # )

    return config

strftime = '%Y-%m-%d %H:%M:%S'

@dataclass
class Job:
    job_id: str
    model_path: Path
    run_path: Path
    params: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None
    nc: Optional[int] = 1
    status: str = "PENDING"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    model_include_patterns: Optional[List[str]] = None
    model_ignore_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", "*.log", ".git","*.h5"])
    retry_count: int = 0
    
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    _log_file: Optional[TextIO] = field(default=None, repr=False)
    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.run_path = Path(self.run_path)
        model_dir,model_filename = self.check_model_path()
        run_dir,run_filename = self.check_run_path()
        self.log_path = self.run_path / 'output.log'
    
    def check_model_path(self):
        model_dir,model_filename = check_path(self.model_path)
        if model_dir is None or model_filename is None:
            raise FileExistsError(f"model_path must point to an existing file: '{self.model_path}'")
        return model_dir,model_filename
            
    def check_run_path(self):
        run_dir,run_filename = check_path(self.run_path)
        if run_filename is None and run_dir is None:
            return run_dir,run_filename
        elif run_dir.exists() and next(run_dir.iterdir(), False):
            raise FileExistsError(f"The run directory is not empty: '{self.run_path}'")
        else:
            return run_dir,run_filename
    
    def mark_started(self) -> None:
        """Call when the process is spawned."""
        self.start_time = time.time()
        self.status = "RUNNING"

    def mark_finished(self, status: str = "COMPLETED") -> None:
        """Call when the process exits or is killed ('COMPLETED', 'KILLED', 'FAILED')."""
        self.end_time = time.time()
        self.status = status

    @property
    def elapsed_time(self) -> float:
        """Calculates running time in seconds. Works for both active and finished jobs."""
        if not self.start_time:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return round(end - self.start_time, 2)

    @classmethod
    def from_config_file(cls, job_id: str, config_path: str | Path) -> "Job":
        """Factory method to create a Job and auto-extract parameters from a JSON file."""
        path = Path(config_path)
        params = {}
        if path.exists() and path.suffix == ".json":
            with open(path, "r") as f:
                params = json.load(f)
        return cls(job_id=job_id, config_path=config_path, params=params)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the job state into a JSON-friendly dict for RPC responses."""
        start_time = datetime.fromtimestamp(self.start_time).strftime(strftime) if self.start_time else None
        end_time = datetime.fromtimestamp(self.end_time).strftime(strftime) if self.end_time else None
        
        return {
            "job_id": self.job_id,
            "model_path": str(self.model_path.resolve()),
            "run_path": str(self.run_path.resolve()),
            "params": self.params,
            "nc": self.nc,
            "status": self.status,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_time": self.elapsed_time,
        }
    

class BaseScheduler(ABC):
    """Core scheduler engine handling background threads, job queues, and process cleanup."""

    def __init__(self, max_concurrent_jobs: int = 2, poll_interval: float = 2.0):
        self.jobs: Dict[str, Any] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_concurrent_cores = 1
        self.poll_interval = poll_interval
        self.max_retries = 3

        self._running = True
        self._monitor_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._monitor_thread.start()

        # Finalizer handles both garbage collection (del) and process exit automatically.
        # No extra atexit registration needed as weakref.finalize handles atexit internally.
        self._finalizer = weakref.finalize(
            self,
            BaseScheduler._finalizer_routine,
            self.jobs,
            self._monitor_thread
        )

    # -------------------------------------------------------------------------
    # Internal Loop & Queue Management
    # -------------------------------------------------------------------------

    @abstractmethod
    def _start_job(self, job: Any) -> None:
        """Child classes MUST implement this method to handle process execution."""
        pass

    def _scheduler_loop(self) -> None:
        """Background loop: polls active jobs and launches pending ones."""
        while self._running:
            try:
                self._update_job_statuses()
                self._process_queue()
            except Exception as e:
                print(f"Error in scheduler loop: {e}")

            time.sleep(self.poll_interval)

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
    def _cleanup_job_proc(job: Any, force: bool = False, timeout: float = 3.0) -> None:
        """Kills the OS process group associated with a single job and releases file handles."""
        proc = getattr(job, "_proc", None)
        
        if proc and proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                if force:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=1.0)
                else:
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        proc.wait(timeout=1.0)
            except (ProcessLookupError, OSError):
                pass
    
        # Ensure log handles are closed and flushed regardless of exit state
        BaseScheduler._cleanup_job_handles(job)

    @staticmethod
    def _cleanup_all_procs(jobs: dict, force: bool = True) -> None:
        """Terminates processes for all active jobs in the dictionary."""
        for job in list(jobs.values()):
            BaseScheduler._cleanup_job_proc(job, force=force)

    @staticmethod
    def _finalizer_routine(jobs: dict, thread: Any) -> None:
        """Static callback executed by weakref.finalize on del or process exit."""
        BaseScheduler._cleanup_all_procs(jobs, force=True)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
            
    def __del__(self):
        


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

@pyro_expose
class DispatchDaemon(Scheduler):
    def __init__(self,run_base_dir=None,max_concurrent_jobs = 2, poll_interval = 2.0):
        super().__init__(max_concurrent_jobs, poll_interval)
        
        self.model_include_patterns = None
        self.model_ignore_patterns = ["__pycache__", "*.pyc", "*.log", ".git","*.h5","*.png"]
        defualt_base_dir = Path.home() / "Documents" / "data"
        self.run_base_dir = Path(run_base_dir) if run_base_dir else defualt_base_dir
        
    @abstractmethod 
    def _start_job(self, job: Job) -> None:
        """Child classes MUST implement this method to handle process execution."""
        pass
    
    def _spawn_run_dir(self, job: str | Job, include_queued: bool = True) -> Path:
        job = self.jobs[job] if isinstance(job, str) else job
        source_dir,_ =  job.check_model_path()
        job.check_run_path()
        run_dir = job.run_path
        job.run_path.mkdir(parents=True, exist_ok=True)
    
        if job.model_include_patterns:
            # Pattern-based copying: Grab only matching files
            for pattern in job.model_include_patterns:
                for source_file in source_dir.glob(pattern):
                    if source_file.is_file():
                        shutil.copy2(source_file, run_dir / source_file.name)
        else:
            # Whole-directory copying: Copy everything except ignored patterns
            ignore_func = shutil.ignore_patterns(*(job.model_ignore_patterns or []))
            shutil.copytree(source_dir, run_dir, dirs_exist_ok=True, ignore=ignore_func)
    
        return run_dir
    
    def _compose_path(self,params, equiv='-', sep='/', order=False,format=None,numDecimals=3):
        run_base_dir = Path(self.run_base_dir)
        tail_path = Path(meta.compose(params,equiv, sep, order,format,numDecimals) )
        return run_base_dir / tail_path
    
    ##########################
    # Job Managment
    ##########################
    @pyro_expose
    def create_job(self, model_path: str,
                   run_path: str = None,
                   job_params: Dict = None,
                   job_config_path: Optional[Path] = None,
                   job_id: Optional[str] = None,
                   nc: Optional[int] = 1,
                   ) -> Dict[str, Any]:
        
        if not run_path and not job_params:
            raise TypeError("Either 'run_path' or 'job_params' must be defined")
        elif run_path and job_params:
            pass
        elif job_params:
            run_path = self._compose_path(job_params)
        
        
        """Creates a job using an optional custom ID or an auto-generated UUID."""
        actual_id = job_id or self._generate_job_id()

        if actual_id in self.jobs:
            raise ValueError(f"Job ID '{actual_id}' already exists.")
        if run_path in self.get_run_paths():
            raise ValueError(f"Run path '{run_path}' already exists in one or more jobs.")
        # job = Job.from_config_file(actual_id, job_config_path)
        job = Job(
            job_id = actual_id, 
            model_path=model_path,
            run_path=run_path,
            params=job_params,
            nc=nc,
            model_include_patterns = self.model_include_patterns,
            model_ignore_patterns = self.model_ignore_patterns,
            )
        self.jobs[actual_id] = job
        return job.to_dict()
    
    @pyro_expose
    def set_job_params(
        self,
        job: str | Job,
        params: Optional[Dict[str, Any]] = None,
        nc: Optional[int] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Updates job parameters and core count.
        
        By default (overwrite=False), new parameters are merged into existing ones.
        Set overwrite=True to wipe existing parameters and replace them completely.
        """
        job = self.jobs[job] if isinstance(job, str) else job
    
        if job.status in ("RUNNING", "COMPLETED", "FAILED", "KILLED"):
            raise ValueError(f"Cannot modify parameters for job in '{job.status}' state.")
    
        # Update core count if provided
        if nc is not None:
            job.nc = nc
    
        # Update or overwrite parameters dictionary
        if params is not None:
            if overwrite:
                job.params = dict(params)
            else:
                job.params.update(params)
    
        return job.to_dict()
    
    @pyro_expose
    def submit_job(self, job: str | Job, nc: [int] = None):
        job = self.jobs[job] if isinstance(job, str) else job
        #### check for issues
        if not job:
            raise ValueError(f"Job {job} not found.")
        if nc:
            job.nc = nc
        job.status = "QUEUED"
        
    @pyro_expose
    def submit_pending(self):
        for job in self.jobs.values():
            if job.status == "PENDING":
                job.status = "QUEUED"
                
    @pyro_expose
    def kill_job(self, *args, **kwargs):
        # Optional pre-processing or logging
        return super().kill_job(*args, **kwargs)
    
    @pyro_expose
    def kill_active_jobs(self, *args, **kwargs):
        return super().kill_active_jobs( *args, **kwargs) 
    
    @pyro_expose
    def clear_jobs(self, include_queued: bool = False) -> List[Dict[str, Any]]:
        """Removes finished jobs (and optionally queued/pending jobs) from memory."""
        self._update_job_statuses()
    
        keys_to_remove = [
            job_id for job_id, job in self.jobs.items()
            if job.status in ("COMPLETED", "FAILED", "KILLED")
            or (include_queued and job.status in ("PENDING", "QUEUED"))
        ]
    
        for job_id in keys_to_remove:
            del self.jobs[job_id]
    
        return self.get_jobs()

    ##########################
    # exposed getters
    ##########################
    @pyro_expose
    def get_jobs(self):
        output = []
        for job in self.jobs.values():
            output.append(job.to_dict())
        return output
    
    @pyro_expose
    def get_run_paths(self):
        output = []
        for job in self.jobs.values():
            output.append(str(job.run_path))
        return output 
    
    @pyro_expose
    def get_ids(self):
        output = []
        for job in self.jobs.values():
            output.append(str(job.job_id))
        return output 
    
    @pyro_expose
    def get_run_base_dir(self):
        return self.run_base_dir
    
    ##########################
    # exposed setters
    ##########################
    
    @pyro_expose
    def set_run_base_dir(self,run_base_dir):
        self.run_base_dir = run_base_dir

    
  
    
@pyro_expose
class WarpXDispatchDaemon(DispatchDaemon):
    def __init__(self,run_base_dir = None, max_concurrent_jobs: int = 2, poll_interval: float = 1.0):
        super().__init__(run_base_dir,max_concurrent_jobs, poll_interval)
        self.model_include_patterns = None
        self.model_ignore_patterns.extend(['Backtrace*','warpx_used_inputs','diags'])
    
    def _start_job(self, job):
        """RPC endpoint to launch the simulation process."""
        
        run_dir = str(self._spawn_run_dir(job))
        
        #### open log file
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        job._log_file = open(job.log_path, "w")
        
        env = self._set_run_env(job)
        #### generate run command
        command = []
        if job.nc>1:
            #command.extend(['mpiexec','-x', 'JOB_CONFIG','-np','%i'%job.nc])
            command.extend(['mpiexec','-np','%i'%job.nc])
        # Daemon handles process spawning and command arguments
        command.extend(['python',job.model_path.name])
        
        #### run job
        job._proc = subprocess.Popen(
            command,
            cwd=run_dir,
            # env=env,
            stdout=job._log_file,
            stderr=subprocess.STDOUT,  # Redirect stderr into stdout
            start_new_session=True     # Decouples process group for clean killing
            )
        job.mark_started()

    def _set_run_env(self, job):
        params = getattr(job, "parameters", getattr(job, "params", {})).copy()
        params["_managed"] = True # flag to tell load_job_config JOB_CONFIG is neccessary
        
        job_config_json = json.dumps(params, default=str)
        return {**os.environ, "JOB_CONFIG": job_config_json}
    
    
    
    

def main(args=None):
    import sys
    from argparse import ArgumentParser
    try:
        import Pyro5.api
    except ImportError:
        print(
            "Error: 'Pyro5' is required to launch the dispatch daemon CLI.\n"
            "Please install it using: pip install Pyro5 (or pip install .[pyro])",
            file=sys.stderr
        )
        sys.exit(1)
    
    parser = ArgumentParser(description="D-Manage proxy dispatch command line launcher.")
    parser.add_argument("-n", "--host", dest="host", default='127.0.0.1', help="hostname to bind server on")
    parser.add_argument("-p", "--port", dest="port", type=int, default=defaultPyroDispatchPort, help="port to bind server on (0=random)")
    options = parser.parse_args(args)

    Pyro5.api.serve(
        {WarpXDispatchDaemon: defaultPyroDispatchName},
        host=options.host,
        port=options.port,
        use_ns=False
    )

if __name__ == "__main__":
    main()