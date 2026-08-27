# -*- coding: utf-8 -*-

import Pyro5.api
#from pathlib import Path
from wxutils.dispatch import WarpXDispatchDaemon
from pywarpx import picmi

constants = picmi.constants

host= '127.0.0.1'
port = 44444
model_path = "/home/marcus/Documents/SimulationProjects/wxutils/tests/test_model/inputs_test_2d_electrostatic_macroscopic_solver_picmi.py"

job_parameters = [{'epsilon0':constants.ep0, 'epsilon1':constants.ep0*10},
                  {'epsilon0':constants.ep0, 'epsilon1':constants.ep0*20}]

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
        job_info_local = dispatch.create_job(model_path, job_params=job_parameter)
    jobs = dispatch.get_jobs()
    job_ids = dispatch.get_ids()
    dispatch.submit_pending()
    # jobs = dispatch.get_jobs()

    
    
    