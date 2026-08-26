# -*- coding: utf-8 -*-

import Pyro5.api
#from pathlib import Path
from wxutils.dispatch import WarpXDispatchDaemon
host= '127.0.0.1'
port = 44444
model_path = "path/to/base_model"

def test_dispatch(self):
    """Make sure factor is running with terminal command 'dmanage-factory'"""
    uri = "PYRO:ProxyDispatch@localhost:44444"
    dispatchProxy =  Pyro5.api.Proxy(uri=uri)
    dispatchProxy.create_job()
    
if __name__ == "__main__":
    uri = "PYRO:ProxyDispatch@localhost:44444"
    dispatchProxy =  Pyro5.api.Proxy(uri=uri)
    job_info_proxy = dispatchProxy.create_job(model_path)
    
    
    dispatchLocal = WarpXDispatchDaemon()
    job_info_local = dispatchLocal.create_job(model_path)
    