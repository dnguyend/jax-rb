# JAX_RB: Riemannian Brownian motion
This package provides a framework to implement Brownian motion on Riemannian manifolds.
## Installation
Requirement: [JAX](https://jax.readthedocs.io/en/latest/installation.html) (pip install jax). If we have access to GPU then install jax cuda following JAX's installation note (for example pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).

To install from git do (assume you have *build*, otherwise do pip install build).

```
pip install git+https://{$GIT_USER}:{$GIT_TOKEN}@github.com/dnguyend/jax-rb.git
```
Remember, if your GIT_USER is an email address, replace @ with %40 in the above, for example, enter
```
pip install git+https://nguyen%40goodwork.org:pw1234@github.com/dnguyend/jax-rb.git
```
if the git username is nguyen@goodwork.org and the token is pw1234. Alternatively, you can download the package and then run pip install locally. If you are on a notebook, we provide a simple form to supply git's user and token, for now. You can generate a token specific to this project.

If you want to to build the project manually from a cloned directory, go to the folder jax-rb then run
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
