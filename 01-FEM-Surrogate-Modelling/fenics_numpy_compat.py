"""Preserve NumPy DOFs in the legacy FEniCS/dlroms conversion path."""
import numpy as np


def install(fe):
    """Patch this process only; never modify installed packages."""
    if int(np.__version__.split('.')[0]) >= 2:
        raise RuntimeError(
            'Legacy FEniCS requires the verified NumPy 1.26.4 runtime here. '
            'Select the project .venv-mixer interpreter and restart the kernel.'
        )
    if getattr(fe.asvector, '_mixer_contiguous', False):
        return
    original = fe.asvector

    def asvector(values, space):
        """Convert tensor or array values to contiguous float64 before the original FEM conversion."""
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        return original(np.ascontiguousarray(values, dtype=np.float64), space)

    asvector._mixer_contiguous = True
    fe.asvector = asvector
    # asfunction in dlroms dispatches through the module's asvector.


def check(fe, space):
    """Verify both strided and contiguous inputs in the actual FEM space."""
    values = np.arange(space.dim() * 3, dtype=float).reshape(-1, 3)[:, 1]
    for data in (values, np.ascontiguousarray(values)):
        actual = fe.asfunction(data, space).vector().get_local()
        np.testing.assert_array_equal(actual, values)
