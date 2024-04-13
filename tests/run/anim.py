import os
import jax
import jax.numpy as jnp
from jax import random, lax
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import rc

from jax_rb.manifolds.sphere import Sphere
from jax_rb.manifolds.se_left_invariant import SELeftInvariant
from jax_rb.manifolds.affine_left_invariant import AffineLeftInvariant

import jax_rb.simulation.matrix_group_integrator as sim
import jax_rb.simulation.global_manifold_integrator as gmi
from jax_rb.utils.utils import grand


def make_brownian_loop(mnf, key, n_dim, n_f, lin_comb, scale, n_pnts):
    """ return one single path
    """
    x_0, key = grand(key, (n_dim, n_f))
    if not (lin_comb is None):
        x_0 = x_0.at[:, n_dim+1:n_f].set(x_0[:, :n_dim+1]@lin_comb)
    x_arr = [10*x_0]

    driver_move, key = grand(key, ((n_dim+1)**2, n_pnts))
    for i in jnp.arange(1, n_pnts):
        g = sim.geodesic_move(mnf, jnp.eye(n_dim+1), driver_move[:, i-1], scale)
        x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
    return jnp.array(x_arr)


def make_brownian(mnf, key, n_dim, n_f, lin_comb, scale, n_pnts):
    """ return one single path
    fori is slower for small n_pnts but fast for large n_pnts
    # the GPU version looks funny. CPU is correct
    """
    x_0, key = grand(key, (n_dim, n_f))

    if not (lin_comb is None):
        x_0 = x_0.at[:, n_dim+1:n_f].set(x_0[:, :n_dim+1]@lin_comb)
    x_arr = jnp.zeros((n_pnts, n_dim*n_f))
    x_arr = x_arr.at[0, :].set(10.*x_0.reshape(-1))
    seq, _ = grand(key, ((n_dim+1)**2, n_pnts))

    def body_fun(i, val):
        g = sim.geodesic_move(mnf, jnp.eye(n_dim+1), seq[:, i-1], scale)
        return val.at[i, :].set((g[:-1, :-1]@val[i-1, :].reshape(n_dim, n_f)
                                 + g[:-1, -1][:, None]).reshape(-1))

    x_arr = lax.fori_loop(1, n_pnts, body_fun, x_arr)
    return x_arr.reshape(n_pnts, n_dim, n_f)


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
            if num > 1:
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

    N = 200
    # x_arr = make_brownian(key)
    x_arr = make_brownian(aff, key, n_dim, n_f, lin_comb, scale, N)

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
    line_ani.save(os.path.join(save_dir, 'af2_animation.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])

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
            if num > 1:
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

    # x_arr = make_brownian(key)
    N = 200
    x_arr = make_brownian(aff, key, n_dim, n_f, lin_comb, scale, N)

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
    line_ani.save(os.path.join(save_dir, 'af3_animation.mp4'),
                  fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


def run_se_3d():
    """Animation for SE(3)
    """
    rc('animation', html='jshtml')
    n_dim = 3
    n_f = 5
    colors = ["r", "b", "g", "m", "c"]

    def func(num, x_arr, lines, dots):
        # ANIMATION FUNCTION
        for i in range(n_f):
            lines[i].set_data(x_arr[:num, :2, i].T)  # cannot set 3d data, break to two commands
            lines[i].set_3d_properties(x_arr[:num, 2, i])
            if num > 1:
                dots[i].set_data(x_arr[num-2:num-1, :2, i].T)
                dots[i].set_3d_properties(x_arr[num-2:num-1, 2, i].T)
        return lines

    # THE DATA POINTS
    key = random.PRNGKey(0)
    se_dim = n_dim*(n_dim+1)//2
    # diag = jnp.ones(se_dim).at[2].set(plot_size/4).at[4].set(plot_size/4).at[5].set(plot_size/4)
    diag = jnp.arange(1, se_dim+1)*3

    se = SELeftInvariant(n_dim, jnp.diag(diag))
    scale = .4

    def make_brownian_(key):
        x_0, key = grand(key, (n_dim, n_f))
        x_arr = [10*x_0]

        for i in jnp.arange(1, N):
            driver_move, key = grand(key, ((n_dim+1)**2,))
            g = sim.geodesic_move(se, jnp.eye(n_dim+1), driver_move, scale)
            x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
        return jnp.array(x_arr)
    N = 200
    lin_comb = jnp.array([[1/2, .5, 0., 0.]]).T

    # x_arr = make_brownian(key)
    x_arr = make_brownian(se, key, n_dim, n_f, lin_comb, scale, N)

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
    line_ani.save(os.path.join(save_dir, 'se3_animation.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])

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
    def make_brownian_(key):
        # x_arr = jnp.empty((N, n_dim, n_f))

        x_0, key = grand(key, (n_dim, n_f))
        x_arr = [10*x_0]

        for i in jnp.arange(1, N):
            driver_move, key = grand(key, ((n_dim+1)**2,))
            g = sim.geodesic_move(se, jnp.eye(n_dim+1), driver_move, scale)
            x_arr.append(g[:-1, :-1]@x_arr[i-1] + g[:-1, -1][:, None])
        return jnp.array(x_arr)

    N = 200
    # x_arr = make_brownian(key)
    x_arr = make_brownian(se, key, n_dim, n_f, None, scale, N)

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
    line_ani.save(os.path.join(save_dir, 'se2_animation.mp4'), fps=30, extra_args=['-vcodec', 'libx264'])

    plt.show()


def sphere_simulate():
    """ long term simulation for the sphere
    """
    n = 3
    n_pnt = 10000
    key = random.PRNGKey(0)
    step = .1
    # x_i = [jnp.zeros(n).at[-1].set(1.)]
    x_i = jnp.zeros((n, n_pnt+1))
    x_i = x_i.at[:, 0].set(jnp.zeros(n).at[-1].set(1.))
    seq, _ = grand(key, (n, n_pnt))
    sph = Sphere(n, 1.)
    ax = plt.axes(projection='3d')

    # for j in range(n_pnt):
    #    x_i.append(gmi.geodesic_move_normalized(sph, x_i[-1], seq[:, j]/jnp.sqrt(jnp.sum(seq[:, j]**2)), step))
    x_i = lax.fori_loop(1, n_pnt+1,
                  lambda i, val: val.at[:, i].set(gmi.geodesic_move_normalized(
                      sph, val[:, i-1], seq[:, i-1]/jnp.sqrt(jnp.sum(seq[:, i-1]**2)), step)), x_i)

    # x_i = jnp.array(x_i).T
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    ax.scatter3D(x_i[0, :], x_i[1, :], x_i[2, :], marker='o', s=.1)
    plt.savefig(os.path.join(save_dir, 'sphere_long_term.png'))


if __name__ == '__main__':
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
    
    print("Please close each animation to move on to the next one.")
    import sys
    if len(sys.argv) < 2:
        print(f"Please run with format python {sys.argv[0]} [output_dir]. Files will be saved in [output_dir]")
        sys.exit()

    save_dir = f"{sys.argv[1]}"
    print(save_dir)
    
    run_se_2d()
    run_se_3d()
    run_affine2d()
    sphere_simulate()
    # 3d works, but is too dense
    # run_affine3d()
