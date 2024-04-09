# More examples using the simulation modules:
## Generating the simulations used in the paper:

The procedure to regenerate the test is to run in the tests/run
```
python run_[manifold].py [output base directory]
```
The full list of runs used to generate the tables in the paper are
run_affine.py, run_glp.py, run_se.py, run_sl.py, run_so.py, run_spd.py, run_sphere.py, run_stiefel.py, run_grassmann.py. For example,
```
python run_se.py ../output
```
Depending on the hardware of your machine, these may take a while. After that, assuming the output is in tests/output, in a test environment with pandas, Jinja2 (for latex) run
```
python collect_all_files.py
```
The script will generate the tables in tex format in the tests/output folder, as well as a large csv file all_sim_detail.csv with detailed output.

## Uniform sampling of compact manifolds.
The following script compare uniform distribution sampling with long time simulation of Riemannian Brownian motion for the sphere, $SO(n)$, Stiefel and Grassmann.
```
python run_uniform.py [output base directory]
```

Finally, anim.py generates animation of Brownian motion for $SE(2$), $SE(3)$ and $Aff^+(2)$ ($Aff^+(3)$ also works but hard to view).

