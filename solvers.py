from dolfin import *


class SolverWithNullSpace(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, MPI.comm_world,
                              PETScKrylovSolver("gmres"), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        as_backend_type(A).set_nullspace(problem.null_space)

        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("ksp_max_it", 1000)
        PETScOptions.set("ksp_gmres_restart", 1000)
        # # PETScOptions.set("ksp_type", "preonly")
        # PETScOptions.set("pc_type", "lu")
        # PETScOptions.set("factor_mat_solver_package", "superlu_dist")
        PETScOptions.set("pc_type", "asm")
        # PETScOptions.set("sub_pc_type", "lu")
        # PETScOptions.set("pc_factor_shift_type", "NONZERO")
        # PETScOptions.set("pc_factor_shift_amount", 1e-6)

        self.linear_solver().set_from_options()


class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, MPI.comm_world,
                              PETScKrylovSolver("gmres"), PETScFactory.instance())

    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        # PETScOptions.set("ksp_type", "gmres")
        # PETScOptions.set("ksp_monitor")
        PETScOptions.set("snes_monitor")
        # PETScOptions.set("ksp_max_it", 1000)
        # PETScOptions.set("ksp_gmres_restart", 200)
        PETScOptions.set("pc_type", "lu")
        # PETScOptions.set("pc_type", "asm")
        # PETScOptions.set("sub_pc_type", "lu")

        self.linear_solver().set_from_options()
