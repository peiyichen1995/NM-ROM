#! /usr/bin/env python
#
from fenics import *
import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from solvers import CustomSolver
from problems import CustomProblem
from scipy.integrate import solve_ivp
import dolfin

dolfin.parameters['linear_algebra_backend'] = 'Eigen'

def burgers_time_viscous(e_num, nu):

    print('')
    print('  Number of elements is %d' % (e_num))
    print('  Viscosity set to %g' % (nu))

    class PeriodicBoundary(SubDomain):

        # Left boundary is "target domain" G
        def inside(self, x, on_boundary):
            return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

        # Map right boundary (H) to left boundary (G)
        def map(self, x, y):
            y[0] = x[0] - 2.0

# Create periodic boundary condition
    pbc = PeriodicBoundary()
#
#  Create a mesh on the interval [0,+1].
#
    x_left = 0.0
    x_right = +2.0
    mesh = IntervalMesh(e_num, x_left, x_right)
#
#  Define the function space to be of Lagrange type
#  using piecewise linear basis functions.
#
    t_num = 1000
    k = 0
    t = 0.0

    t_plot = 0.0
    t_final = 0.5

    V = FunctionSpace(mesh, "CG", 1)
    snapshots = np.zeros((e_num + 1, t_num + 1))
#
#  Define the boundary conditions.
   # if X <= XLEFT + eps, then U = U_LEFT
   # if X_RIGHT - eps <= X, then U = U_RIGHT

    u_left = +1.0

    def on_left(x, on_boundary):
        return (on_boundary and near(x[0], x_left))
    bc_left = DirichletBC(V, u_left, on_left)

    u_right = +1.0

    def on_right(x, on_boundary):
        return (on_boundary and near(x[0], x_right))
    bc_right = DirichletBC(V, u_right, on_right)

    bc = [bc_left, bc_right]
    # bc = []

    # Sub domain for Periodic boundary condition


#
#  Define the initial condition.
#
    # u_init = Expression ( "x[0]", degree = 1 )
    u_init = Expression(
        "x[0] < 1 ? 1+A*(sin(2*pi*x[0]-pi/2)+1) : 1", degree=1, A=nu / 2)
    # u_init = Expression ( "1+A*(sin(10*pi*x[0]-pi/2)+1)", degree = 1, A = nu/2 )

#
#  Define the trial functions (u) and test functions (v).
#
    u = Function(V)
    u_old = Function(V)
    v = TestFunction(V)
#
#  Set U and U0 by interpolation.
#
    u.interpolate(u_init)
    u_old.assign(u)
#
#  Set the time step.
#  We need a UFL version "DT" for the function F,
#  and a Python version "dt" to do a conditional in the time loop.
#

    DT = Constant(t_final / t_num)
    dt = t_final / t_num
#
#  Set the source term.
#
    f = Expression("0.0", degree=0)
#
#  Write the function to be satisfied.
#
    n = FacetNormal(mesh)
#
#  Write the function F.
#
    # F = \
    # ( \
    #   dot ( u - u_old, v ) / DT \
    # + nu * inner ( grad ( u ), grad ( v ) ) \
    # + inner ( u * u.dx(0), v ) \
    # - dot ( f, v ) \
    # ) * dx

    F = \
        (
            dot(u - u_old, v) / DT
            + inner(u * u.dx(0), v)
            - dot(f, v)
        ) * dx
#
#  Specify the jacobian.
#
    J = derivative(F, u)
#
#  Do the time integration.
#

    while (True):

        if (k % 25 == 0):
            # arr = u.vector().get_local()
            # snapshots[:,k] = arr
            # plot ( u, title = ( 'burgers time viscous %g' % ( t ) ) )
            plt.plot(V.tabulate_dof_coordinates(), u.vector().get_local())
            plt.grid(True)
            filename = ('output/burgers_time_viscous_%d.png' % (k))
            plt.savefig(filename)
            print('Graphics saved as "%s"' % (filename))
            plt.close()
            t_plot = t_plot + 0.1

        if (t_final <= t - dt):
            print('')
            print('Reached final time.')
            break

        arr = u.vector().get_local()
        snapshots[:, k] = arr

        k = k + 1
        t = t + dt

        # solve ( F == 0, u, bc, J = J )
        problem = NonlinearVariationalProblem(F, u, bc, J)
        solver = NonlinearVariationalSolver(problem)
        solver.solve()

        u_old.assign(u)

    return snapshots


def pod(snapshots, nu, e_num):
    podmodes, svals, _ = spla.svd(snapshots, full_matrices=False)
    filename = ('output/burgers_viscous_%g.png' % (nu))
    plt.semilogy(svals, '.')
    plt.grid(True)
    plt.title('$\mu = $' + str(nu))
    plt.savefig(filename)
    plt.close()

    # find pod dimension
    err_tol = 1e-5
    poddim = 1
    err = 1 - np.sum(svals[:poddim]) / np.sum(svals)
    while (err > err_tol):
        poddim += 1
        err = 1 - np.sum(svals[:poddim]) / np.sum(svals)

    print('POD dimension: ' + str(poddim))

    selected_podmodes = podmodes[:, :poddim]

    x_left = 0.0
    x_right = +2.0
    mesh = IntervalMesh(e_num, x_left, x_right)

    t_num = 1000
    k = 0
    t = 0.0

    t_plot = 0.0
    t_final = 0.5

    timegrid = np.linspace(t, t_final, t_num)

    V = FunctionSpace(mesh, "CG", 1)

    #  Define the initial condition.
    u_init = Expression(
        "x[0] < 1 ? 1+A*(sin(2*pi*x[0]-pi/2)+1) : 1", degree=1, A=nu / 2)

    #
    #  Define the trial functions (u) and test functions (v).
    #
    u = Function(V)
    u_old = Function(V)
    v = TestFunction(V)

    #
    #  Set U and U0 by interpolation.
    #
    u.interpolate(u_init)
    u_old.assign(u)

    #
    #  Set the time step.
    #  We need a UFL version "DT" for the function F,
    #  and a Python version "dt" to do a conditional in the time loop.
    #

    DT = Constant(t_final / t_num)
    dt = t_final / t_num
    #
    #  Set the source term.
    #
    f = Expression("0.0", degree=0)
    #
    #  Write the function to be satisfied.
    #
    n = FacetNormal(mesh)
    #
    #  Write the function F.
    #

    F = \
        (
            dot(u - u_old, v) / DT
            + inner(u * u.dx(0), v)
            - dot(f, v)
        ) * dx
    #
    #  Specify the jacobian.
    #
    J = derivative(F, u)

    J_mat = assemble(J)
    J_array = J_mat.array()
    J_red = np.matmul(selected_podmodes.T, J_array)
    J_red = np.matmul(J_red, selected_podmodes)

    def burgers_nonl_func(ufun):
        cform = inner(ufun.dx(0) * ufun, v) * dx
        cass = assemble(cform)
        return cass

    # as a vector to vector map

    def burgers_nonl_vec(uvec):
        ufun = Function(V)
        ufun.vector().set_local(uvec)
        bnlform = burgers_nonl_func(ufun)
        bnlvec = bnlform.get_local()
        return bnlvec

    def redbrhs(time, redvec):
        inflatedv = selected_podmodes.dot(redvec)
        redconv = selected_podmodes.T.dot(burgers_nonl_vec(inflatedv))
        # return -J_red.dot(redvec) - redconv.flatten()
        return - redconv.flatten()


    u = u.vector().get_local()
    # u = np.matmul(selected_podmodes.T, u)
    u = selected_podmodes.T.dot(u)

    redburgsol = solve_ivp(redbrhs, (t, t_final), u,
                           t_eval=timegrid, method='RK23')


    ks = [25*i for i in range(1,41)]
    for k in ks:
        plt.plot(V.tabulate_dof_coordinates(), selected_podmodes.dot(redburgsol.y[:,k-1]))
        plt.grid(True)
        filename = ('output/reduced_burgers_time_viscous_%d.png' % (k))
        plt.savefig(filename)
        plt.close()

    return redburgsol


def burgers_time_viscous_test():

    # *****************************************************************************80
    #
    # burgers_time_viscous_test tests burgers_time_viscous.
    #
    #  Licensing:
    #
    #    This code is distributed under the GNU LGPL license.
    #
    #  Modified:
    #
    #    21 October 2018
    #
    #  Author:
    #
    #    John Burkardt
    #
    import time

    print(time.ctime(time.time()))
#
#  Report level = only warnings or higher.
#
    # level = 30
    # set_log_level ( level )

    # print ( '' )
    # print ( 'burgers_time_viscous_test:' )
    # print ( '  FENICS/Python version' )
    # print ( '  Solve the time-dependent 1d viscous Burgers equation.' )

    e_num = 1000
    # nu = 0.05
    # nus = [1/10**i for i in range(11)]
    nus = [1.0]
    for nu in nus:
        print('nu = %g' % (nu))
        snapshots = burgers_time_viscous(e_num, nu)
        # svd
        solution = pod(snapshots, nu, e_num)
        solution = solution.y
#
#  Terminate.
#
    # print ( "" )
    # print ( "burgers_time_viscous_test:" )
    # print ( "  Normal end of execution." )
    # print ( '' )
    # print ( time.ctime ( time.time() ) )

    return


if (__name__ == '__main__'):

    burgers_time_viscous_test()
