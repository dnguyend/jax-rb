import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc

from jax_rb.manifolds.se_left_invariant import SELeftInvariant
from jax_rb.manifolds.affine_left_invariant import AffineLeftInvariant

import jax_rb.simulation.matrix_group_integrator as sim
from jax_rb.utils.utils import grand


def run_affine2d():
    """Animation for :math:`Aff^+(2)`
    """
    rc('animation', html='jshtml')
    n_dim = 2
    n_f = 4
    colors = ["r", "b", "m", "g"]

    def func(num, x_arr, lines, dots):
        # ANIMATION FUNCTION
        for i in range(n_f):
            lines[i].set_data(x_arr[:num, :, i].T)  # cannot set 3d data, break to two commands
            if num > 0:
                dots[i].set_data(x_arr[num-2:num-1, :2, i].T)
        return lines

    # THE DATA POINTS
    key = random.PRNGKey(0)
    af_dim = n_dim*(n_dim+1)
    # diag = jnp.ones(se_dim).at[2].set(plot_size/4).at[4].set(plot_size/4).at[5].set(plot_size/4)
    diag = jnp.arange(af_dim)*70. + 10

    aff = AffineLeftInvariant(n_dim, jnp.diag(diag))
    scale = .5

    # lin_comb = jnp.array([[.25, .25, 0.25, .25], [1/2, .5, 0., 0.]]).T
    lin_comb = jnp.array([[1/2, .5, 0.]]).T

    def make_brownian(key):
        x_0, key = grand(key, (n_dim, n_f))
        x_0 = x_0.at[:, n_dim+1:n_f].set(x_0[:, :n_dim+1]@lin_comb)
        x_arr = [10*x_0]

        for i in jnp.arange(1, N):
            driver_move, key = grand(key, ((n_dim+1)**2,))
            g = sim.geodesic_move(aff, jnp.eye(n_dim+1), driver_move, scale)
            x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
        return jnp.array(x_arr)
    N = 200
    x_arr = make_brownian(key)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = plt.axes()
    lines = [plt.plot(x_arr[:, 0, i], x_arr[:, 1, i],
                      c=colors[i])[0]
             for i in range(n_f)]  # For line plot

    dots = [plt.plot(x_arr[:1, 0, i], x_arr[0, 1, i],
                     c=colors[i], marker='o')[0]
             for i in range(n_f)]

    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_title('Trajectory of Riemannian Brownian motion Aff(2)')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, func, frames=N, fargs=(x_arr, lines, dots), interval=50, blit=False)
    line_ani.save('af2_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


def run_affine3d():
    """Animation for :math:`Aff(3)`
    """
    rc('animation', html='jshtml')
    n_dim = 3
    n_f = 6
    colors = ["r", "b", "c", "m", "g", "y"]

    def func(num, x_arr, lines, dots):
        # ANIMATION FUNCTION
        for i in range(n_f):
            lines[i].set_data(x_arr[:num, :2, i].T)  # cannot set 3d data, break to two commands
            lines[i].set_3d_properties(x_arr[:num, 2, i])
            if num > 0:
                dots[i].set_data(x_arr[num-2:num-1, :2, i].T)
                dots[i].set_3d_properties(x_arr[num-2:num-1, 2, i].T)
        return lines

    # THE DATA POINTS
    key = random.PRNGKey(0)
    af_dim = n_dim*(n_dim+1)
    # diag = jnp.ones(se_dim).at[2].set(plot_size/4).at[4].set(plot_size/4).at[5].set(plot_size/4)
    diag = jnp.arange(1, af_dim+1)*10.

    aff = AffineLeftInvariant(n_dim, jnp.diag(diag))
    scale = .5

    # lin_comb = jnp.array([[.25, .25, 0.25, .25], [1/2, .5, 0., 0.]]).T
    lin_comb = jnp.array([[1/2, .5, 0., 0.]]).T

    def make_brownian(key):
        x_0, key = grand(key, (n_dim, n_f))
        x_0 = x_0.at[:, n_dim+1:n_f].set(x_0[:, :n_dim+1]@lin_comb)
        x_arr = [10*x_0]

        for i in jnp.arange(1, N):
            driver_move, key = grand(key, ((n_dim+1)**2,))
            g = sim.geodesic_move(aff, jnp.eye(n_dim+1), driver_move, scale)
            x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
        return jnp.array(x_arr)
    N = 200
    x_arr = make_brownian(key)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    lines = [plt.plot(x_arr[:, 0, i], x_arr[:, 1, i],
                      x_arr[:, 2, i], c=colors[i])[0]
             for i in range(n_f)]  # For line plot

    dots = [plt.plot(x_arr[:1, 0, i], x_arr[0, 1, i],
                     x_arr[:1, 2, i],
                     c=colors[i], marker='o')[0]
             for i in range(n_f)]

    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('Z(t)')
    ax.set_title('Trajectory of Riemannian Brownian motion Aff(3)')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, func, frames=N, fargs=(x_arr, lines, dots), interval=50, blit=False)
    line_ani.save('af3_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


def run_se_3d():
    """Animation for SE(3)
    """
    rc('animation', html='jshtml')
    n_dim = 3
    n_f = 4
    colors = ["r", "b", "g", "m"]

    def func(num, x_arr, lines, dots):
        # ANIMATION FUNCTION
        for i in range(n_f):
            lines[i].set_data(x_arr[:num, :2, i].T)  # cannot set 3d data, break to two commands
            lines[i].set_3d_properties(x_arr[:num, 2, i])
            if num > 0:
                dots[i].set_data(x_arr[num-1, :2, i].T)
                dots[i].set_3d_properties(x_arr[num-1, 2, i].T)
        return lines

    # THE DATA POINTS
    key = random.PRNGKey(0)
    se_dim = n_dim*(n_dim+1)//2
    # diag = jnp.ones(se_dim).at[2].set(plot_size/4).at[4].set(plot_size/4).at[5].set(plot_size/4)
    diag = jnp.arange(1, se_dim+1)*3

    se = SELeftInvariant(n_dim, jnp.diag(diag))
    scale = .4

    def make_brownian(key):
        x_0, key = grand(key, (n_dim, n_f))
        x_arr = [10*x_0]

        for i in jnp.arange(1, N):
            driver_move, key = grand(key, ((n_dim+1)**2,))
            g = sim.geodesic_move(se, jnp.eye(n_dim+1), driver_move, scale)
            x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
        return jnp.array(x_arr)
    N = 200
    x_arr = make_brownian(key)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    lines = [plt.plot(x_arr[:, 0, i], x_arr[:, 1, i],
                      x_arr[:, 2, i], c=colors[i])[0]
             for i in range(n_f)]  # For line plot

    dots = [plt.plot(x_arr[0, 0, i], x_arr[0, 1, i],
                     x_arr[0, 2, i],
                     c=colors[i], marker='o')[0]
             for i in range(n_f)]

    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_zlabel('Z(t)')
    ax.set_title('Trajectory of Riemannian Brownian motion SE(3)')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, func, frames=N, fargs=(x_arr, lines, dots), interval=50, blit=False)
    line_ani.save('se3_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


def run_se_2d():
    rc('animation', html='jshtml')
    n_dim = 2
    n_f = 4
    plot_size = 10
    colors = ["r", "b", "g", "m"]

    def func(num, x_arr, lines, dots):
        # ANIMATION FUNCTION
        for i in range(n_f):
            lines[i].set_data(x_arr[:num, :, i].T)
            if num > 0:
                dots[i].set_data(x_arr[num-2:num-1, :, i].T)
        return lines

    key = random.PRNGKey(0)
    se_dim = n_dim*(n_dim+1)//2
    diag = jnp.ones(se_dim).at[0].set(plot_size/4)
    se = SELeftInvariant(n_dim, jnp.diag(diag))
    scale = .4
    def make_brownian(key):
        # x_arr = jnp.empty((N, n_dim, n_f))

        x_0, key = grand(key, (n_dim, n_f))
        x_arr = [10*x_0]

        for i in jnp.arange(1, N):
            driver_move, key = grand(key, ((n_dim+1)**2,))
            g = sim.geodesic_move(se, jnp.eye(n_dim+1), driver_move, scale)
            x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
        return jnp.array(x_arr)

    N = 200
    x_arr = make_brownian(key)

    # GET SOME MATPLOTLIB OBJECTS
    fig = plt.figure()
    ax = plt.axes()
    lines = [plt.plot(x_arr[:, 0, i], x_arr[:, 1, i],  c=colors[i])[0]
             for i in range(n_f)]  # For line plot

    dots = [plt.plot(x_arr[0, 0, i], x_arr[0, 1, i],  c=colors[i], marker='o')[0]
             for i in range(n_f)]
    ax.set_xlabel('X(t)')
    ax.set_ylabel('Y(t)')
    ax.set_title('Trajectory of Riemannian Brownian motion SE(2)')

    # Creating the Animation object
    line_ani = animation.FuncAnimation(
        fig, func, frames=N, fargs=(x_arr, lines, dots,), interval=200, blit=False)
    line_ani.save('se2_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


    

def run_affine_bad():
    """ Animation for :math:`Aff^+(3)`
    """
    rc('animation', html='jshtml')
    fig = plt.figure(figsize=(8, 6))
    n_dim = 3
    
    ax = plt.axes(projection='3d')
    key = random.PRNGKey(0)
    colors = itertools.cycle(["r", "b", "g"])
    r = 40
    plot_size = 1000
    
    n_f = 5
    
    x_pnt, key = grand(key, (n_dim, n_f))
    # x_pnt = jnp.zeros((n_dim, n_f))
    # x_pnt = x_pnt

    af_dim = n_dim*(n_dim+1)
    # se_n = sem.SELeftInvariant(n_dim, jnp.eye(se_dim).at[-1].set(7))
    
    # diag = jnp.ones(se_dim).at[2].set(r/10).at[4].set(r/10).at[5].set(r/10)
    
    af_n = AffineLeftInvariant(n_dim, .1*jnp.eye(af_dim))
    scale = .01

    def frame(_):
        ax.clear()
     
        driver_move, key = grand(key, ((n_dim+1)**2,))
        g = sim.geodesic_move(af_n, jnp.eye(n_dim+1), driver_move, scale)
        x_pnt = g[:-1, :-1]@x_pnt + g[:-1, -1][:, None]
        plt.title("Brownian Motion")
        ax.set_xlabel('X(t)')
        ax.set_xlim3d(-plot_size, plot_size)
        ax.set_ylabel('Y(t)')
        ax.set_ylim3d(-plot_size, plot_size)
        ax.set_zlabel('Z(t)')
        ax.set_zlim3d(-plot_size, plot_size)
        plot=ax.scatter3D(x_pnt[0, :], x_pnt[1, :], x_pnt[2, :], c=next(colors))
        return plot

    anim = animation.FuncAnimation(fig, frame, frames=100, blit=False, repeat=True)
    plt.show()

    
if __name__ == '__main__':
    print("Please close each animation to move on to the next one.")
    run_se_2d()    
    run_se_3d()
    run_affine2d()
    # 3d works, but is too dense
    # run_affine3d()
