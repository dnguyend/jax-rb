# JAX_RB: Riemannian Brownian motion
This package provides a framework to implement Brownian motion on Riemannian manifolds.
## Installation
Requirement: [jax](https://jax.readthedocs.io/en/latest/installation.html) (pip install jax). If we have access to GPU then install jax cuda following jax installation note (for example pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html).

To install from git do

```
pip install git+ssh://git@github.com/dnguyend/jax-rb.git
```

Eventually we will upload this to pypi and the users can install from there.
## Examples
* Cookbook.
* Simulation for the manifolds in the article are in the folder tests/run. See the README under the tests/run folder.

## Tests
* See the README file in the test folder for testing the connection, projection, Laplace-Beltrami operator and the drift.
