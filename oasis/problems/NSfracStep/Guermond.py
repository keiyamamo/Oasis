from __future__ import print_function
from math import degrees

from numpy.lib.arraysetops import union1d
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import numpy as np

"""
This script is based on TaylorGreen2D.py but modified to reproduce the results from guermond(2006) papaer.
The diffrenece from TaylorGreen is as follows. 

1. Domain. In this case, x and y span from -1 to 1 and nx, ny are set to 48.
2. Analytical solution. Please look at equation (3.30) in the paper.
3. Boundary condition. Dirichlet boundary condition for the velocity. Nothing for the pressure
"""

# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):
    ua0 = " pi*sin(t_e)*sin(2*pi*x[1])*sin(pi*x[0])*sin(pi*x[0]) "
    ua1 = " -pi*sin(t_e)*sin(2*pi*x[0])*sin(pi*x[1])*sin(pi*x[1]) "
    pa = " sin(t_e)*cos(pi*x[0])*sin(pi*x[1]) "
    NS_parameters.update(
        ua0 = ua0,
        ua1 = ua1,
        pa = pa,
        nu=0.01,
        T=1.,
        dt=0.001,
        Nx=24, Ny=24, # N = 48 in Guermond, but first try with smaller N
        folder="guermond_results",
        plot_interval=100,
        save_step=100,
        checkpoint=100,
        print_intermediate_info=100,
        compute_error=100,
        use_krylov_solvers=True,
        velocity_degree=1,
        pressure_degree=1,
        krylov_report=False)

    NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
                                       'report': False,
                                       'relative_tolerance': 1e-12,
                                       'absolute_tolerance': 1e-12}
    NS_expressions.update(dict(
        initial_fields=dict(
            #constrained_domain=PeriodicDomain(),
            u0=ua0,  
            u1=ua1, 
            p=pa), 
        dpdx=('-pi*sin(t)*sin(pi*x[0])*sin(pi*x[1])',              # modified
              'pi*sin(t)*cos(pi*x[0]*cos(pi*x[1]))'),              # modified
        total_error=zeros(3)))
    
    
    v_degree = NS_parameters["velocity_degree"]
    p_degree = NS_parameters["pressure_degree"]

    NS_expressions.update(
        ue0 = Expression(ua0, degree=v_degree, t_e=0),
        ue1 = Expression(ua1, degree=v_degree, t_e=0),
        pe = Expression(pa, degree=p_degree, t_e=0))

# Positions of points are modified
def mesh(Nx, Ny, **params):
    return RectangleMesh(Point(-1, -1), Point(1, 1), Nx, Ny)

# added, DiricheltBC for the velocity
def create_bcs(V, NS_expressions, **NS_namespace):
    return dict(u0=[DirichletBC(V, NS_expressions["ue0"] , "on_boundary")],
                u1=[DirichletBC(V, NS_expressions["ue1"], "on_boundary")],
                p =[]
                )
# source term 
def body_force(t, ua0, ua1, pa, mesh, **NS_namespace):
    x = SpatialCoordinate(mesh)
    u_vec = as_vector([eval(ua0), eval(ua1)])
    p_ = eval(pa)
    f = diff(u_vec, t) - div(grad(u_vec)) + grad(p_)
    return f

# nothing is changed from TaylorGreen2D
def initialize(q_, q_1, q_2, VV, t, nu, dt, initial_fields, **NS_namespace):
    """Initialize solution.

    Use t=dt/2 for pressure since pressure is computed in between timesteps.

    """
    for ui in q_:
        if 'IPCS' in NS_parameters['solver']:
            deltat = dt / 2. if ui is 'p' else 0.
        else:
            deltat = 0.
        vv = interpolate(Expression((initial_fields[ui]),
                                     element=VV[ui].ufl_element(),
                                     t=t + deltat, nu=nu), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = vv.vector()[:]
            deltat = -dt
            vv = interpolate(Expression((initial_fields[ui]),
                                        element=VV[ui].ufl_element(),
                                        t=t + deltat, nu=nu), VV[ui])
            q_2[ui].vector()[:] = vv.vector()[:]
    q_1['p'].vector()[:] = q_['p'].vector()[:]


def start_timestep_hook(q_, t, V, bcs, NS_expressions, **NS_namesapce):
     # update t in the analytical solution
    NS_expressions["ue0"].t = t
    NS_expressions["ue1"].t = t
    # update the boundary condition 
    bcs2 = create_bcs(V, NS_expressions)
    bcs['u0'] = bcs2['u0']
    bcs['u1'] = bcs2['u1']
    # apply updated boundary condition 
    for ui in q_:
          [bc.apply(q_[ui].vector()) for bc in bcs[ui]]
    

def temporal_hook(q_, t, nu, VV, dt, plot_interval, initial_fields, tstep, sys_comp,
                  compute_error, total_error, **NS_namespace):
    """Function called at end of timestep.

    Plot solution and compute error by comparing to analytical solution.
    Remember pressure is computed in between timesteps.

    """
   
    if tstep % plot_interval == 0:
        plot(q_['u0'], title='u')
        plot(q_['u1'], title='v')
        plot(q_['p'], title='p')
        
    
    if tstep % compute_error == 0:
        err = {}
        for i, ui in enumerate(sys_comp):
            if 'IPCS' in NS_parameters['solver']:
                deltat_ = dt / 2. if ui is 'p' else 0.
            else:
                deltat_ = 0.
            ue = Expression((initial_fields[ui]),
                            element=VV[ui].ufl_element(),
                            t=t - deltat_, nu=nu)
            ue = interpolate(ue, VV[ui])
            uen = norm(ue.vector())
            ue.vector().axpy(-1, q_[ui].vector())
            # from IPython import embed;embed();import sys; sys.exit(1)
            error = norm(ue.vector()) / uen # somehow error becomes 1, meaning that norm(ue.vector()) and uen has the same value
            err[ui] = "{0:2.6e}".format(norm(ue.vector()) / uen)
            total_error[i] += error * dt
        if MPI.rank(MPI.comm_world) == 0:
            print("Error is ", err, " at time = ", t)
        


# added some codes for plotting differene between  analytical solution and computed solution 
def theend_hook(mesh, q_, t, dt, nu, VV, sys_comp, total_error, initial_fields, **NS_namespace):
    final_error = zeros(len(sys_comp))
    
    for i, ui in enumerate(sys_comp):
        if 'IPCS' in NS_parameters['solver']:
            deltat = dt / 2. if ui is 'p' else 0.
        else:
            deltat = 0.
        ue = Expression((initial_fields[ui]),
                        element=VV[ui].ufl_element(),
                        t=t - deltat, nu=nu)
        ue = interpolate(ue, VV[ui])
        final_error[i] = errornorm(q_[ui], ue)

    #plot the error between analytical solution and computed solution of the pressure  
    x0 = np.linspace(-1, 1, 49)
    x1 = np.linspace(-1, 1, 49)
    X0, X1 = np.meshgrid(x0, x1)    
    
    if ui is 'p':
        ue.vector().axpy(-1, q_[ui].vector())
        arr = ue.vector().get_local()
        Arr = np.reshape(arr, (49, 49))
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X0, X1, Arr, cmap=cm.coolwarm)
        plt.show()
    else:
        pass

    hmin = mesh.hmin()
    if MPI.rank(MPI.comm_world) == 0:
        print("hmin = {}".format(hmin))
    s0 = "Total Error:"
    s1 = "Final Error:"
    for i, ui in enumerate(sys_comp):
        s0 += " {0:}={1:2.6e}".format(ui, total_error[i])
        s1 += " {0:}={1:2.6e}".format(ui, final_error[i])

    if MPI.rank(MPI.comm_world) == 0:
        print(s0)
        print(s1)
