jax\_rb.manifolds
=================

.. automodule:: jax_rb.manifolds

	Global Manifold
	===============
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.global_manifold
	   :members:		
   
        Sphere
	===============
	.. autosummary::
	   :toctree: _autosummary

        .. automodule:: jax_rb.manifolds.sphere
	.. autoclass:: Sphere
   
	Symmetric Positive Definite Matrix Manifold
	============================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.spd
	.. autoclass:: PositiveDefiniteManifold

	Stiefel Manifold
	=================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.stiefel
	.. autoclass:: RealStiefelAlpha
   
	Grassmann Manifold
	==========================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.grassmann
	.. autoclass:: Grassmann

	Matrix Lie Group Left Invariant Metric
	=======================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.matrix_left_invariant
	   :members:

	   Required implementations
	   =========================
	   .. autofunction:: jax_rb.manifolds.matrix_left_invariant.MatrixLeftInvariant._lie_algebra_proj
	   .. autofunction:: jax_rb.manifolds.matrix_left_invariant.MatrixLeftInvariant._mat_apply
		      

           Base class
	   ===========			     
   
	:math:`\mathrm{GL}^+(n)` Generalized Linear Group Positive Determinant
	========================================================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.glp_left_invariant
	.. autoclass:: GLpLeftInvariant

	:math:`\mathrm{Aff}^+(n)` Affine Linear Group Positive Determinant
	====================================================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.affine_left_invariant
	.. autoclass:: AffineLeftInvariant
   
	:math:`\mathrm{SL}(n)` Special Linear Group
	===============================================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.sl_left_invariant
	.. autoclass:: SLLeftInvariant
   
	:math:`\mathrm{SO}(n)` Special Orthogonal Group
	===============================================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.so_left_invariant
	.. autoclass:: SOLeftInvariant
   
	:math:`\mathrm{SE}(n)` Special Euclidean Group
	===============================================================
	.. autosummary::
	   :toctree: _autosummary

	.. automodule:: jax_rb.manifolds.se_left_invariant
	.. autoclass:: SELeftInvariant

   
   
   



