# -*- coding: utf-8 -*-
try:
    import Pyro5.api
    #from Pyro5.server import is_private_attribute
    import Pyro5.serializers
except ImportError:
    raise ImportError("Module 'Pyro5' must be installed to use the rpc package, use 'pip install dmanage[Pyro5]'")

import uuid
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess

defaultPyroDispatchHost = "localhost"
defaultPyroDispatchPort = 44444
defaultPyroDispatchName = "ProxyDispatch"

ONLY_EXPOSED = True
Pyro5.api.config.PICKLE_ENABLE = False

@dataclass
class Job:
    job_id: str
    model_path: Path
    run_path: Path = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    config_path: Optional[Path] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __post_init__(self):
        self.model_path = Path(self.model_path)
    
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

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Extract a specific simulation parameter."""
        return self.parameters.get(key, default)

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
        return {
            "job_id": self.job_id,
            "file_path": str(self.model_path.resolve()),
            "parameters": self.parameters,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time": self.elapsed_time,
        }

@Pyro5.api.expose
class DispatchDaemon:
    def __init__(self):
        # Maps job_id -> Job metadata object
        self.jobs: Dict[str, Job] = {}
        # Private map: job_id -> live OS process handle (never exposed via RPC)
        self._processes: Dict[str, subprocess.Popen] = {}
    
    def _generate_job_id(self, prefix: str = "job") -> str:
        """Generates a short, unique ID (e.g., job_a1b2c3d4)."""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @Pyro5.api.expose
    def create_job(self, model_path: str,
                   job_parameters: Optional[Dict] = None,
                   job_config_path: Optional[Path] = None,
                   job_id: Optional[str] = None,
                   ) -> Dict[str, Any]:
        """Creates a job using an optional custom ID or an auto-generated UUID."""
        actual_id = job_id or self._generate_job_id()

        if actual_id in self.jobs:
            raise ValueError(f"Job ID '{actual_id}' already exists.")

        # job = Job.from_config_file(actual_id, job_config_path)
        job = Job(job_id = actual_id, model_path=model_path)
        self.jobs[actual_id] = job
        return job.to_dict()
    
    @Pyro5.api.expose
    def list_jobs(self):
        return self.jobs
    
    def start_job(self, job_id: str) -> Dict[str, Any]:
        """RPC endpoint to launch the simulation process."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found.")

        # Daemon handles process spawning and command arguments
        proc = subprocess.Popen(
            ["warpx", "-i", str(job.file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self._processes[job_id] = proc
        job.mark_started()
        return job.to_dict()

    def kill_job(self, job_id: str) -> Dict[str, Any]:
        """RPC endpoint to terminate a simulation."""
        proc = self._processes.get(job_id)
        if proc and proc.poll() is None:  # Process is still running
            proc.terminate()
            proc.wait()
            del self._processes[job_id]
            
        job = self.jobs[job_id]
        job.mark_finished(status="KILLED")
        return job.to_dict()
    
@Pyro5.api.expose
class WarpXDispatchDaemon(DispatchDaemon):
    pass
    


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
    