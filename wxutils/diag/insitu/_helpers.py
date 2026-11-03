# -*- coding: utf-8 -*-


import matplotlib as mpl
def move_figure(f,pos):
    """Move figure's upper left corner to pixel (x, y)"""
    x,y = pos[0],pos[1]
    backend = mpl.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
  
    
def update_home_decorator(f):
    def newf(*args, **kwargs):
        ret = f(*args, **kwargs)
        
        toolbar = f.__self__
        fig = toolbar.canvas.figure
        
        is_at_home = False
        if hasattr(toolbar, '_nav_stack'):
            elements = toolbar._nav_stack._elements
            if not elements or toolbar._nav_stack() == elements[0]:
                is_at_home = True
                
        if is_at_home:
            for ax in fig.axes:
                # Reset old step data limits
                ax.dataLim.ignore(True)
                
                # 1. Recalculate 1D line plot limits
                ax.relim()
                
                # 2. Recalculate scatter plot / collection limits
                for c in ax.collections:
                    ax.update_datalim(c.get_datalim(ax.transData))
                
                # 3. Recalculate 2D image (imshow) limits
                for im in ax.images:
                    extent = im.get_extent()
                    ax.set_xlim(extent[0], extent[1])
                    ax.set_ylim(extent[2], extent[3])
                
                ax.autoscale(enable=True, axis='both')
                ax.autoscale_view()
            
            fig.canvas.draw_idle()
            
        return ret
    return newf