# JAX_RB: Riemannian Brownian motion
This package provides a framework to implement Brownian motion on Riemannian manifolds.
## Installation
Requirement: [jax](https://jax.readthedocs.io/en/latest/installation.html) (pip install jax). If we have access to GPU then install jax cuda following jax installation note (for example pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).

To install from git do (assume you have build, otherwise do pip install build).

```
pip install git+https://{$GIT_USER}:{$GIT_PWD}@github.com/dnguyend/jax-rb.git
```
Or download then run pip install locally. If you are on a notebook, we provide a simple form to supply user and password to git, for now. 

If you are not on a notebook and want to use pip, to build the project manually, go to the folder jax-rb then run
```
python -m build
```
assuming you have jax installed.

To build the document, go to the jax-rb/docs folder then run 
```
make html.
```
Then go to the html, under jax-rb/docs/_build/html/index.html and open that file in your browser, - that will allow to navigate the documentation
Eventually we will upload this to pypi and the users can install from there.
## Examples
* [Cookbook](https://github.com/dnguyend/jax-rb/blob/main/examples/JAX_RB_Cookbook.ipynb).
* Simulation for the manifolds in the article are in the folder tests/run. See the README under the tests/run folder.

## Tests
* See the README file in the folder 'tests' for testing the connection, projection, Laplace-Beltrami operator and the drift.
