# Detailed testing of the projection, the Levi-Civita connection, and the Laplace-Beltrami operator.

* The notebook notebooks/SE_RB.ipynb contains the detailed testing for the manifold $SE(n)$.
* For other manifold, simpy run the scripts test_[manifold].py in the folder.

* To test the HeatKernel, look at notebooks/test_heat_kernel.ipynb. You may need to manually load tests/utils.heat_kernels.py to your colab workspace and run pip install mpmath. Alternatively in an environment with numpy and mpmath run

```
PYTHONPATH=., python sphere_heat_kernel.py
```

in the tests folder, also assuming mpmath is installed. The reason that utils.heat_kernels is not in the main utils folder is we do not want to introduce the dependency of jax_rb on mpmath.

* Include tests for Langevin processes on [groups](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/LangevinGroup.ipynb), [Stiefel manifolds](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/LangevinStiefel.ipynb) and [hypersurfaces](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/TestRetractiveIntegrator.ipynb). Also show example of [Cayley retraction on SO(n)](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/TestExpmAndCayleyIntegrator.ipynb)
