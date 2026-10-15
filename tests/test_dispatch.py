# -*- coding: utf-8 -*-

try:
    import Pyro5.api
except ImportError:
    # No-op decorator if Pyro5 is not installed
    pass

#from pathlib import Path
from wxutils.dispatch import WarpXDispatcher
from pywarpx import picmi
from pathlib import Path

constants = picmi.constants

host= '127.0.0.1'
port = 44444
script_dir = Path(__file__).parent.resolve()
model_path = script_dir / "test_model/test_model.py"
job_parameters = [{'epsilon0':constants.ep0, 'epsilon1':constants.ep0*10},
                  {'epsilon0':constants.ep0, 'epsilon1':constants.ep0*20}]

def test_dispatch(self):
    """Make sure factor is running with terminal command 'dmanage-factory'"""
    uri = "PYRO:ProxyDispatch@localhost:44444"
    dispatchProxy =  Pyro5.api.Proxy(uri=uri)
    dispatchProxy.create_job()
    
if __name__ == "__main__":
    uri = "PYRO:ProxyDispatch@localhost:44444"
    # dispatch =  Pyro5.api.Proxy(uri=uri)
    
    dispatch = WarpXDispatcher()
    dispatch.set_poll_interval(1.0)
    for job_parameter in job_parameters:
        job_info_local = dispatch.create_job(model_path, job_params=job_parameter,nc=2)
    jobs = dispatch.get_jobs()
    job_ids = dispatch.get_ids()
    dispatch.submit_pending()
    # jobs = dispatch.get_jobs()
    # dispatch.kill_active_jobs()
    # dispatch.stop_scheduler()
    
    
    