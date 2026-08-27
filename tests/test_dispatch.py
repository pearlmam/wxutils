# -*- coding: utf-8 -*-

import Pyro5.api
#from pathlib import Path
from wxutils.dispatch import WarpXDispatchDaemon
host= '127.0.0.1'
port = 44444
model_path = "/home/marcus/Documents/SimulationProjects/wxutils/tests/test_model/inputs_test_2d_electrostatic_macroscopic_solver_picmi.py"

job_parameters = [{'param1':'100', 'param2':'200'},
                  {'param1':'100', 'param2':'300'}]

def test_dispatch(self):
    """Make sure factor is running with terminal command 'dmanage-factory'"""
    uri = "PYRO:ProxyDispatch@localhost:44444"
    dispatchProxy =  Pyro5.api.Proxy(uri=uri)
    dispatchProxy.create_job()
    
if __name__ == "__main__":
    uri = "PYRO:ProxyDispatch@localhost:44444"
    #dispatch =  Pyro5.api.Proxy(uri=uri)

    dispatch = WarpXDispatchDaemon()
    for job_parameter in job_parameters:
        job_info_local = dispatch.create_job(model_path, job_parameters=job_parameter)
    jobs = dispatch.get_jobs()
    job_ids = dispatch.get_ids()
    dispatch._spawn_run_dir(list(dispatch.jobs.values())[0])
    # for job_id in job_ids:
    #     dispatch.submit_job(job_id,nc=2)
    # jobs = dispatch.get_jobs()
    
    # status = dispatch.poll_job(job_ids[1])
    
    
    