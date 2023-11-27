import timesteppers
import finite
from timesteppers import StateVector
from scipy import sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
import numpy as np
from farray import apply_matrix

import matplotlib.pyplot as plt

def plot_2D(matrix):
    lim_margin = -0.05
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot()
    I, J = matrix.shape
    matrix_mag = np.log10(np.abs(matrix))
    ax.pcolor(matrix_mag[::-1])
    ax.set_xlim(-lim_margin, I+lim_margin)
    ax.set_ylim(-lim_margin, J+lim_margin)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()

class DiffusionXBC:
    
    def __init__(self, c, D, d2x, spatial_order, grid):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        M = sparse.eye(N, N)
        M = M.tocsr()
        M[0, :] = 0
        M[-1,:] = 0
        M.eliminate_zeros()
        self.M = M
        
        L = -D*d2x.matrix
        L = L.tocsr()
        L[0,:] = 0
        L[-1,:] = 0
        L[0, 0] = 1

        # the 1 at the end comes from the fact we want a first derivative
        dof = spatial_order + 1

        dof = dof - (1 - dof % 2)

        prob_i = np.arange(dof)[:, None]
        prob_j = np.arange(dof) - dof + 1
        S = 1/factorial(prob_i)*(prob_j*grid.dx)**prob_i

        b = np.zeros(dof)
        b[1] = 1

        new_stencil = np.linalg.solve(S, b)

        L[-1, -dof:] = new_stencil
        # L[-1, -1] = -1/grid.dx
        L.eliminate_zeros()
        self.L = L


class DiffusionYBC:
    
    def __init__(self, c, D, d2y):
        self.X = StateVector([c], axis=1)
        N = c.shape[1]
        self.M = sparse.eye(N, N)
        self.L = -D*d2y.matrix


class DiffusionBC:

    def __init__(self, c, D, spatial_order, domain):
        self.t = 0
        self.iter = 0

        self.X = timesteppers.StateVector([c])

        d2x = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], axis=0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], axis=1)

        diffx = DiffusionXBC(c, D, d2x, spatial_order, domain.grids[0])
        diffy = DiffusionYBC(c, D, d2y)
        self.diffusion_ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.diffusion_ts_y = timesteppers.CrankNicolson(diffy, 1)

    def step(self, dt):
        self.diffusion_ts_y.step(dt/2)
        self.diffusion_ts_x.step(dt)
        self.diffusion_ts_y.step(dt/2)

        self.t += dt
        self.iter += 1


class Wave2DBC:

    def __init__(self, u, v, p, spatial_order, domain):
        self.X = StateVector([u, v, p])
        N = len(u)
        dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], axis=0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], axis=1)
        def F(X):
            X.scatter()

            r1 = -(dx @ X.variables[2])
            r2 = -(dy @ X.variables[2])
            r3 = -(dx @ X.variables[0]) - (dy @ X.variables[1])

            return np.vstack((r1, r2, r3))
        self.F = F
        def BC(X):
            X.data[0, :] = 0
            X.data[N - 1, :] = 0
        self.BC = BC


class DiffusionX:
    
    def __init__(self, c, D, d2x):
        self.X = StateVector([c], axis=0)
        N = c.shape[0]
        self.M = sparse.eye(N, N)
        self.L = -D*d2x.matrix


class DiffusionY:
    
    def __init__(self, c, D, d2y):
        self.X = StateVector([c], axis=1)
        N = c.shape[1]
        self.M = sparse.eye(N, N)
        self.L = -D*d2y.matrix


class Reaction:

    def __init__(self, c):
        self.X = StateVector([c])
        self.F = lambda x: x.data * (1 - x.data)


class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.t = 0
        self.iter = 0

        diffx = DiffusionX(c, D, dx2)
        diffy = DiffusionY(c, D, dy2)
        self.diffusion_ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.diffusion_ts_y = timesteppers.CrankNicolson(diffy, 1)

        reac = Reaction(c)
        self.reaction_ts = timesteppers.RK22(reac)

    def step(self, dt):
        self.reaction_ts.step(dt/2)
        # things will go wrong if you do not have the split operators next
        # to each other, so they need to be in the middle
        self.diffusion_ts_y.step(dt)
        self.diffusion_ts_x.step(dt)
        self.reaction_ts.step(dt/2)
        self.t += dt
        self.iter += 1


class VBDiffusionX:
    
    def __init__(self, u, v, nu, d2x):
        self.X = StateVector([u, v], axis=0)
        N = u.shape[0]
        N_full = u.shape[0] + v.shape[0]
        self.M = sparse.eye(N_full, N_full)

        Z = sparse.csr_matrix((N, N))
        base_L = sparse.bmat([[d2x.matrix, Z],
                              [Z, d2x.matrix]])
        self.L = -nu*base_L


class VBDiffusionY:
    
    def __init__(self, u, v, nu, d2y):
        self.X = StateVector([u, v], axis=1)
        N = u.shape[1]
        N_full = u.shape[1] + v.shape[1]
        self.M = sparse.eye(N_full, N_full)

        Z = sparse.csr_matrix((N, N))
        base_L = sparse.bmat([[d2y.matrix, Z],
                              [Z, d2y.matrix]])
        self.L = -nu*base_L


class AdvectionX:

    def __init__(self, u, v, dx, dy):
        self.X = StateVector([u])

        self.F = lambda X: -X.data * (dx @ X.data) -v * (dy @ X.data)


class AdvectionY:

    def __init__(self, u, v, dx, dy):
        self.X = StateVector([v])

        self.F = lambda X: -u * (dx @ X.data) -X.data * (dy @ X.data)


class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        self.t = 0
        self.iter = 0

        dx0 = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)
        dy0 = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)
        dx1 = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], 0)
        dy1 = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], 1)

        dx2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], 0)
        dy2 = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], 1)

        diffx = VBDiffusionX(u, v, nu, dx2)
        diffy = VBDiffusionY(u, v, nu, dy2)
        self.diffusion_ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.diffusion_ts_y = timesteppers.CrankNicolson(diffy, 1)

        adv_x = AdvectionX(u, v, dx0, dy0)
        adv_y = AdvectionY(u, v, dx1, dy1)
        self.advection_ts_x = timesteppers.RK22(adv_x)
        self.advection_ts_y = timesteppers.RK22(adv_y)

    def step(self, dt):
        self.advection_ts_y.step(dt/2)
        self.advection_ts_x.step(dt/2)
        self.diffusion_ts_y.step(dt)
        self.diffusion_ts_x.step(dt)
        self.advection_ts_x.step(dt/2)
        self.advection_ts_y.step(dt/2)

        self.t += dt
        self.iter += 1


class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = None
        if isinstance(rho0, np.ndarray):
            M00 = np.diag(rho0)
        else:
            M00 = rho0 * I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = d.matrix
        L10 = None
        if isinstance(gammap0, np.ndarray):
            L10 = gammap0 @ d.matrix
        else:
            L10 = gammap0 * d.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        
        N = len(c)
        self.M = sparse.eye(N, N)
        self.L = -D*d2.matrix
        
        f = lambda X: X.data*(c_target - X.data)
        
        self.F = f


class ReactionDiffusionFI:
    
    def __init__(self, c, D, spatial_order, grid):
        self.X = timesteppers.StateVector([c])
        d2 = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(c)
        I = sparse.eye(self.N)
        
        self.M = I
        self.L = -D*d2.matrix

        def F(X):
            return X.data*(1-X.data)
        self.F = F
        
        def J(X):
            c_matrix = sparse.diags(X.data)
            return sparse.eye(self.N) - 2*c_matrix
        
        self.J = J


class BurgersFI:
    
    def __init__(self, u, nu, spatial_order, grid):
        self.X = timesteppers.StateVector([u])
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid)
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = len(u)
        I = sparse.eye(self.N)
        self.M = I
        self.L = -nu*d2x.matrix

        def F(X):
            return -X.data * (dx.matrix @ X.data)
        self.F = F
        
        def J(X):
            tiled = np.tile(X.data, (self.N, 1)).T
            # tiled = (X.data * np.ones((self.N, self.N))).T

            return -dx.matrix.multiply(tiled) - sparse.diags(dx.matrix @ X.data)
        self.J = J


class ReactionTwoSpeciesDiffusion:
    
    def __init__(self, X, D, r, spatial_order, grid):
        self.X = X
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid)
        self.N = self.X.N
        print(self.N)
        print(self.X.data.shape)
        I = sparse.eye(self.N)
        Z = sparse.csr_matrix((self.N, self.N))

        self.M = sparse.bmat([[I, Z],
                              [Z, I]])

        self.L = sparse.bmat([[-D*d2x.matrix, Z],
                              [Z, -D*d2x.matrix]])

        eq1_mul1 = sparse.bmat([[I, Z],
                                [Z, Z]])
        eq1_mul2 = sparse.bmat([[Z, I],
                                [Z, Z]])
        eq2_mul1 = sparse.bmat([[Z, Z],
                                [I, Z]])
        eq2_mul2 = sparse.bmat([[Z, Z],
                                [Z, I]])

        def F(X):
            c1_eq1 = eq1_mul1 @ X.data
            c2_eq1 = eq1_mul2 @ X.data
            c1_eq2 = eq2_mul1 @ X.data
            c2_eq2 = eq2_mul2 @ X.data
            return (c1_eq1 - c1_eq1 * c1_eq1 - c1_eq1 * c2_eq1) + r * (c2_eq2 * c1_eq2 - c2_eq2 * c2_eq2)
        self.F = F
        
        def J(X):
            X.scatter()
            c1 = X.variables[0]
            c2 = X.variables[1]

            d11 = sparse.diags(1 - 2 * c1 - c2)
            d12 = sparse.diags(-c1)
            d21 = sparse.diags(r * c2)
            d22 = sparse.diags(r * c1 - 2 * r * c2)

            return sparse.bmat([[d11, d12],
                                [d21, d22]])
        self.J = J


class NavierStokes:
    def __init__(self, u, v, p, mu, kappa, gamma, rho, spatial_order, domain):
        self.X = timesteppers.StateVector([u, v, p])
        dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], axis=0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], axis=1)

        d2x = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], axis=0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], axis=1)

        self.N = len(u)

        I = sparse.eye(self.N)
        Z = sparse.csr_matrix((self.N, self.N))

        self.M = sparse.bmat([[I, Z, Z],
                              [Z, I, Z],
                              [Z, Z, I]])
        
        self.L = (1/rho) * sparse.bmat([[-mu * d2x.matrix, -mu * d2y.matrix, dx.matrix],
                                        [-mu * d2x.matrix, -mu * d2y.matrix, dy.matrix],
                                        [Z, Z, -kappa * (d2x.matrix + d2y.matrix)]])

        def F(X):
            X.scatter()
            u = X.variables[0]
            v = X.variables[1]
            p = X.variables[2]

            r1 = -u * (dx @ u)
            r2 = -v * (dy @ v)
            r3 = -(u * (dx @ p) + v * (dy @ p)) - gamma * p * (dx @ u + dy @ v)

            return np.vstack((r1, r2, r3))
        self.F = F

        def J(X):
            X.scatter()
            u = X.variables[0]
            v = X.variables[1]
            p = X.variables[2]

            # t1 = np.tile(u, (self.N, 1)).T
            # print(t1.shape)
            m1 = dx.matrix.multiply(u)
            r1 = -m1 - dx @ u #sparse.diags(dx @ u)

            # t2 = np.tile(v, (self.N, 1)).T
            m2 = dy.matrix.multiply(v)
            r2 = -m2 - dy @ v # sparse.diags(dy @ v)

            r31 = -(dx @ p) - gamma * p * m1
            r32 = -(dy @ p) - gamma * p * m2

            # t3 = np.tile(p, (self.N, 1)).T

            m3 = dx.matrix.multiply(p)
            m4 = dy.matrix.multiply(p)

            r33 = - (u * m3 + v * m4) - gamma * (dx @ u + dy @ v)

            # d11 = sparse.diags(1 - 2 * c1 - c2)
            # d12 = sparse.diags(-c1)
            # d21 = sparse.diags(r * c2)
            # d22 = sparse.diags(r * c1 - 2 * r * c2)

            return sparse.bmat([[r1, Z, Z],
                                [Z, r2, Z],
                                [r31, r32, r33]])
        self.J = J

class IncompressibleNavierStokes:
    def __init__(self, u, v, p, nu, spatial_order, domain):
        self.t = 0
        self.iter = 0

        self.u = u
        self.v = v
        self.p = p
        self.X = timesteppers.StateVector([u, v, p])
        self.dx = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[0], axis=0)
        self.dy = finite.DifferenceUniformGrid(1, spatial_order, domain.grids[1], axis=1)

        self.d2x = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[0], axis=0)
        self.d2y = finite.DifferenceUniformGrid(2, spatial_order, domain.grids[1], axis=1)

        self.N = len(u)

        diffx = VBDiffusionX(u, v, nu, self.d2x)
        diffy = VBDiffusionY(u, v, nu, self.d2y)
        self.diffusion_ts_x = timesteppers.CrankNicolson(diffx, 0)
        self.diffusion_ts_y = timesteppers.CrankNicolson(diffy, 1)

        adv_x = AdvectionX(u, v, self.dx, self.dy)
        adv_y = AdvectionY(u, v, self.dx, self.dy)
        self.advection_ts_x = timesteppers.RK22(adv_x)
        self.advection_ts_y = timesteppers.RK22(adv_y)

        # I = sparse.eye(self.N)
        # Z = sparse.csr_matrix((self.N, self.N))

        # self.M = sparse.bmat([[I, Z, Z],
        #                       [Z, I, Z],
        #                       [Z, Z, I]])
        
        # self.L = sparse.bmat([[Z, Z, d2x.matrix],
        #                       [Z, Z, d2y.matrix],
        #                       [Z, Z, Z]])

        # self.diffusion_ts_y = timesteppers.CrankNicolson(diffy, 1)

    def step(self, dt):
        # self.diffusion_ts_y.step(dt/2)
        # self.diffusion_ts_x.step(dt)
        # self.diffusion_ts_y.step(dt/2)

        d = (self.dx @ self.u + self.dy @ self.v)
        # print(f"MAX: {np.max(d)}")
        # plot_2D()

        self.advection_ts_y.step(dt/2)
        self.advection_ts_x.step(dt/2)
        self.diffusion_ts_y.step(dt)
        self.diffusion_ts_x.step(dt)
        self.advection_ts_x.step(dt/2)
        self.advection_ts_y.step(dt/2)

        # u and v are currently U*
        LHS = (self.d2x.matrix + self.d2y.matrix)
        LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

        # print(f"LHS: {np.min(LHS)}, {np.max(LHS)}")
        
        RHS = -(self.dx @ self.u + self.dy @ self.v)
        # plot_2D(self.dy.matrix.toarray())

        # print(f"RHS: {np.min(RHS)}, {np.max(RHS)}")
        p = LU.solve(RHS)
        # plot_2D(p)

        # print(f"p mean: {np.mean(p)}")
        # print(f"p max: {np.max(p)}")
        # plot_2D(p - np.mean(p))
        self.p[:] = p
        # p[0,:] = 0
        # p[-1,:] = 0
        # p[:, -1] = 0
        # p[:, 0] = 0

        # print(f"p max: {np.max(p)}")

        self.u += self.dx @ p
        self.v += self.dy @ p

        # print(np.max(self.u))
        # print(np.max(self.v))

        # print(self.u[95:110,:])

        self.t += dt
        self.iter += 1
