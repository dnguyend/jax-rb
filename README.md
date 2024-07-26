# JAX_RB: Riemannian Brownian motion
This package provides a framework to implement Brownian motion on Riemannian manifolds. The theory is developed in [this paper](https://arxiv.org/abs/2406.02879).
Check out the [documentation page](https://dnguyend.github.io/jax-rb/index.html).
## WHY:
* Simulation of SDEs on manifolds.
* Simulation of [Riemannian Brownian motions](https://github.com/dnguyend/jax-rb/blob/main/examples/JAX_RB_Cookbook.ipynb) and [Riemannian uniform distributions](https://github.com/dnguyend/jax-rb/blob/main/tests/run/run_uniform.py).
* Simulation of [Riemannian Langevin processes](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/LangevinGroup.ipynb) and [sampling distributions on manifolds with given densities](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/LangevinStiefel.ipynb).
## WHAT:
* Several [example manifolds](https://dnguyend.github.io/jax-rb/manifolds.html) with flexible choices of metrics.
* Several [simulation schemes](https://dnguyend.github.io/jax-rb/simulation.html).
* Extendable to new manifolds/new SDEs.

## HOW:
## Installation
Requirement: [JAX](https://jax.readthedocs.io/en/latest/installation.html) (pip install jax). If we have access to GPU then install jax cuda following JAX's installation note (for example pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).

To install from git do (assume you have *build*, otherwise do pip install build).

```
pip install git+https://github.com/dnguyend/jax-rb
```
Alternatively, you can clone the project to your local directory then add the directory to your PYTHONPATH. View [an example](https://github.com/dnguyend/jax-rb/blob/main/tests/notebooks/test_heat_kernel.ipynb) using sys.append, you can use PYTHONPATH similarly.

If you want to build the project manually from a cloned directory, go to the folder jax-rb then run
```
python -m build
```
assuming you have JAX installed.

To build the document, you need to install sphinx (pip install sphinx, pip install sphinx-rtd-theme) then go to the jax-rb/docs folder and run 
```
make html.
```
After that, go to the index.html file under jax-rb/docs/_build/html, and open that file in your browser, - which will allow to navigate the documentation.
Eventually, we will upload this to pypi and the users can install the library from there.
## Examples
* [Cookbook](https://github.com/dnguyend/jax-rb/blob/main/examples/JAX_RB_Cookbook.ipynb).
* Simulations for the manifolds in the article are in the folder [tests/run](https://github.com/dnguyend/jax-rb/tree/main/tests/run). See the [README](https://github.com/dnguyend/jax-rb/tree/main/tests/run/README.md) under that folder.

## Tests
* See the README file in the folder [tests](https://github.com/dnguyend/jax-rb/tree/main/tests) for testing the connection, projection, Laplace-Beltrami operator and the drift. The folder [notebooks](https://github.com/dnguyend/jax-rb/tree/main/tests/notebooks) also contains tests of the heat kernels.
