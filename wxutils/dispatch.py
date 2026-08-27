# -*- coding: utf-8 -*-
try:
    import Pyro5.api
    #from Pyro5.server import is_private_attribute
    import Pyro5.serializers
except ImportError:
    raise ImportError("Module 'Pyro5' must be installed to use the rpc package, use 'pip install dmanage[Pyro5]'")

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

strftime = '%Y-%m-%d %H:%M:%S'

@dataclass
class Job:
    job_id: str
    model_path: Path
    run_path: Path
    parameters: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None
    nc: Optional[int] = 1
    status: str = "PENDING"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    model_include_patterns: Optional[List[str]] = None
    model_ignore_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", "*.log", ".git","*.h5"])
    
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
        return cls(job_id=job_id, config_path=config_path, parameters=params)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the job state into a JSON-friendly dict for RPC responses."""
        start_time = datetime.fromtimestamp(self.start_time).strftime(strftime) if self.start_time else None
        end_time = datetime.fromtimestamp(self.end_time).strftime(strftime) if self.end_time else None
        
        return {
            "job_id": self.job_id,
            "model_path": str(self.model_path.resolve()),
            "run_path": str(self.run_path.resolve()),
            "parameters": self.parameters,
            "nc": self.nc,
            "status": self.status,
            "start_time": start_time,
            "end_time": end_time,
            "elapsed_time": self.elapsed_time,
        }


class Scheduler:
    def __init__(self, max_concurrent_jobs: int = 2, poll_interval: float = 1.0):
        self.jobs: Dict[str, Job] = {}
        self.max_concurrent_jobs = max_concurrent_jobs
        self.poll_interval = poll_interval
        
        # Shutdown flag for background thread
        self._running = True
        
        # Start background monitoring thread
        self._monitor_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._monitor_thread.start()
    

    def _start_job(self, job_id: str, nc: [int] = None) -> Dict[str, Any]:
        """RPC endpoint to launch the simulation process."""
        raise NotImplementedError("start_job must be set by child.")
        
    def _scheduler_loop(self) -> None:
        """Background loop: polls active jobs and launches pending ones."""
        while self._running:
            try:
                self._update_job_statuses()
                self._process_queue()
            except Exception as e:
                print(f"Error in scheduler loop: {e}")
            
            time.sleep(self.poll_interval)
        
    def _poll_job(self, job):
        if job.status == "RUNNING" and job._proc:
            retcode = job._proc.poll()
            if retcode is not None:
                job.exit_code = retcode
                job.end_time = time.time()
                job.status = "COMPLETED" if retcode == 0 else "FAILED"
                self._cleanup_job_handles(job)
        return job.to_dict()
    
    def _update_job_statuses(self) -> None:
        """Polls all active jobs to update status and free finished slots."""
        for job in list(self.jobs.values()):
            self._poll_job(job)
                    
    def _process_queue(self) -> None:
        """Launches queued jobs if slot capacity allows."""
        running_jobs = [j for j in self.jobs.values() if j.status == "RUNNING"]
        pending_jobs = [j for j in self.jobs.values() if j.status == "QUEUED"]

        available_slots = self.max_concurrent_jobs - len(running_jobs)

        # Launch pending jobs up to the concurrency limit
        for job in pending_jobs[:available_slots]:
            self._start_job(job)
            
            
    def _spawn_run_dir(self, job: Job) -> Path:
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
        
    def _cleanup_job_handles(self, job: Job) -> None:
        if job._log_file and not job._log_file.closed:
            job._log_file.close()
        job._proc = None
        
    def _generate_job_id(self, prefix: str = "job") -> str:
        """Generates a short, unique ID (e.g., job_a1b2c3d4)."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _compose_path(self,parameters, equiv='-', sep='/', order=False,format=None,numDecimals=3):
        run_base_dir = Path(self.run_base_dir)
        tail_path = Path(meta.compose(parameters,equiv, sep, order,format,numDecimals) )
        return run_base_dir / tail_path


@Pyro5.api.expose
class DispatchDaemon(Scheduler):
    def __init__(self,run_base_dir=None,max_concurrent_jobs = 2, poll_interval = 1.0):
        super().__init__(max_concurrent_jobs, poll_interval)
        
        self.model_include_patterns = None
        self.model_ignore_patterns = ["__pycache__", "*.pyc", "*.log", ".git","*.h5","*.png"]
        defualt_base_dir = Path.home() / "Documents" / "data"
        self.run_base_dir = Path(run_base_dir) if run_base_dir else defualt_base_dir
    
    def _start_job(self, job):
        """RPC endpoint to launch the simulation process."""
        raise NotImplementedError("start_job must be set by child.")
    
    @Pyro5.api.expose
    def create_job(self, model_path: str,
                   run_path: str = None,
                   job_parameters: Dict = None,
                   job_config_path: Optional[Path] = None,
                   job_id: Optional[str] = None,
                   ) -> Dict[str, Any]:
        
        if not run_path and not job_parameters:
            raise TypeError("Either 'run_path' or 'job_parameters' must be defined")
        if run_path and job_parameters:
            pass
        if job_parameters:
            run_path = self._compose_path(job_parameters)
        
        
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
            parameters=job_parameters,
            model_include_patterns = self.model_include_patterns,
            model_ignore_patterns = self.model_ignore_patterns,
            )
        self.jobs[actual_id] = job
        return job.to_dict()
    
    @Pyro5.api.expose
    def submit_job(self, job_id: str, nc: [int] = None):
        job = self.jobs.get(job_id)
        
        #### check for issues
        if not job:
            raise ValueError(f"Job {job_id} not found.")
        if nc:
            job.nc = nc
        job.status = "QUEUED"
        
    @Pyro5.api.expose
    def submit_pending(self):
        for job in self.jobs.values():
            if job.status == "PENDING":
                job.status = "QUEUED"
        
    @Pyro5.api.expose
    def kill_job(self, job_id: str) -> Dict[str, Any]:
        job = self.jobs[job_id]
        if job.status == "RUNNING" and job._proc:
            # Terminate the process group (parent mpiexec + worker threads)
            os.killpg(os.getpgid(job._proc.pid), signal.SIGTERM)
            job._proc.wait()
            job.mark_finished("KILLED")
            self._cleanup_job_handles(job)
            
        return job.to_dict()
    
    ##########################
    # exposed setters
    ##########################
    @Pyro5.api.expose
    def get_jobs(self):
        output = []
        for job in self.jobs.values():
            output.append(job.to_dict())
        return output
    
    @Pyro5.api.expose
    def get_run_paths(self):
        output = []
        for job in self.jobs.values():
            output.append(str(job.run_path))
        return output 
    
    @Pyro5.api.expose
    def get_ids(self):
        output = []
        for job in self.jobs.values():
            output.append(str(job.job_id))
        return output 
    
    
    ##########################
    # exposed setters
    ##########################
    
    @Pyro5.api.expose
    def set_run_base_dir(self,run_base_dir):
        self.run_base_dir = run_base_dir

    
  
    
@Pyro5.api.expose
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
        
        #### generate run command
        command = []
        if job.nc>1:
            command.extend(['mpiexec','-np','%i'%job.nc])
        # Daemon handles process spawning and command arguments
        command.extend(['python',job.model_path.name])
        
        #### run job
        job._proc = subprocess.Popen(
            command,
            cwd=run_dir,
            stdout=job._log_file,
            stderr=subprocess.STDOUT,  # Redirect stderr into stdout
            start_new_session=True     # Decouples process group for clean killing
            )
        job.mark_started()
    

def main(args=None):
    from argparse import ArgumentParser
    parser = ArgumentParser(description="D-Manage proxy dispatch command line launcher.")
    parser.add_argument("-n", "--host", dest="host",default='127.0.0.1', help="hostname to bind server on")
    parser.add_argument("-p", "--port", dest="port", type=int,default=defaultPyroDispatchPort, help="port to bind server on (0=random)")
    #parser.add_argument("--use_ns", dest="use_ns", type=bool,default=False, help="to use a NameServer or not")
    options = parser.parse_args(args)
    Pyro5.api.serve({WarpXDispatchDaemon: defaultPyroDispatchName},host=options.host,
                    port=options.port, use_ns=False)
    
if __name__ == "__main__":
    main()
    