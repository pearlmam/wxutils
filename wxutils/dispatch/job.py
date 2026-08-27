# -*- coding: utf-8 -*-

import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List,TextIO
import subprocess

try:
    import Pyro5.api
    pyro_expose = Pyro5.api.expose
    pyro_behavior = Pyro5.server.behavior
except ImportError:
    # No-op decorator if Pyro5 is not installed
    def pyro_expose(obj):
        return obj



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