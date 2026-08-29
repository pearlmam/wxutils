# -*- coding: utf-8 -*-
  
from abc import ABC, abstractmethod  
from typing import Optional, Dict, Any, List
import inspect
import json
from pathlib import Path
import subprocess
import os
import shutil
try:
    import dmanage.metadata.metastring as meta
except:
    meta = None
    
    
from .scheduler import Scheduler
from .job import Job, pyro_expose,pyro_behavior

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

@pyro_expose
@pyro_behavior(instance_mode="single")
class DispatchDaemon(Scheduler,ABC):
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
        if meta:
            tail_path = Path(meta.compose(params,equiv, sep, order,format,numDecimals) )
        else:
            raise ModuleNotFoundError("package dmanage is required to automatically compose paths, you must specify the 'run_path' for each job.")
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
@pyro_behavior(instance_mode="single")
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
            # command.extend(['mpiexec','-x', 'JOB_CONFIG','-np','%i'%job.nc])
            command.extend(['mpiexec','-np','%i'%job.nc])
        command.extend(['python',job.model_path.name])
        
        #### run job
        env = self._set_run_env(job)   # to pass arguments to simulation
        job._proc = subprocess.Popen(
            command,
            cwd=run_dir,
            env=env,
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
        defaultPyroDispatchHost = "localhost"
        defaultPyroDispatchPort = 44444
        defaultPyroDispatchName = "ProxyDispatch"

        Pyro5.api.config.PICKLE_ENABLE = False
    except ImportError:
        print(
            "Error: 'Pyro5' is required to launch the dispatch daemon CLI.\n"
            "Please install it using: pip install Pyro5 (or pip install .[pyro])",
            file=sys.stderr
        )
        sys.exit(1)
    
    parser = ArgumentParser(description="D-Manage proxy dispatch command line launcher.")
    parser.add_argument("-n", "--host", dest="host", default=defaultPyroDispatchHost, help="hostname to bind server on")
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