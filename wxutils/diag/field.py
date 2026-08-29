# -*- coding: utf-8 -*-


class Field():
    def __init__(self, name, io,save_period,**kw):
        super().__init__(name, io,save_period,**kw)
        self.suffix_format = "%06T"
        self.name = name
        self.period = period
        self.lev = 0
        
        
        
    def pre_initialize(self,sim):
        callbacks.installcallback("beforeInitEsolve", self.post_initialize)
        callbacks.installafterstep(self.save)
        callbacks.installcallback("onbreaksignal",self.close)
        self.sim = sim
        self.series = opmd.Series(self.path, opmd.Access.create)
        self.series.author = getpass.getuser()
        self.series.set_attribute("dependencies", f"{self.series.software} {self.series.software_version}")
        self.series.set_software('wxutils',version("wxutils"))
    
    def post_initialize(self):
        self.xp, _ = load_cupy()
        self.grid_info()
        self.field_data = self.sim.fields.get(self.name,level=self.lev)
        os.makedirs(self.path.parent,exist_ok=True)
        with open(self.path.parent/"paraview.pmd", "w") as f:
            f.write(f"openpmd_{self.suffix_format}.bp5\n")
        
    def save(self):
        step = self.sim.extension.warpx.getistep(self.lev)
        if step % self.period == 0:
            data = self.field_data[...]
            shape = list(data.shape)
            # shape.append(1)
            data = data.reshape(shape[0], 1, shape[1])
            shape = data.shape
            step = self.sim.extension.warpx.getistep(self.lev)
            t = self.sim.extension.warpx.gett_new(self.lev)
            
            it = self.series.iterations[step]
            #print(dir(it))
            it.set_time(t)
            it.set_dt(self.sim.time_step_size)
            it.set_time_unit_SI(1.0)
            # it.set_attribute("data_time",t)
            mesh = it.meshes[self.name]
            mesh.grid_spacing = [self.dxyz[0],1.0,self.dxyz[1]]
            mesh.grid_global_offset = [0., 0., 0.]
            mesh.axis_labels = ["x","y","z"]
            mesh.data_order = 'C'
            component = mesh[opmd.Mesh.SCALAR]
            #print(f"dxz={mesh.grid_spacing}, offset = {mesh.grid_global_offset},shape = {data.shape}")
            component.reset_dataset(opmd.Dataset(data.dtype, shape))
            component[:, :] = data
            it.close()
            self.series.flush()
    
    def close(self,):
        self.series.flush()
        self.series.close()