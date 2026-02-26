"""Crud-aware calibration test — thin wrapper around code.crud_test.

This top-level convenience module re-exports everything from the inner
``code.crud_test`` package so that ``python crud_test.py`` works from
the repository root.  It exists for backward compatibility and CLI
convenience; library consumers should prefer ``from code import ...``.

CLI usage::

    python scripts/crud_test.py --data data/cache/cached_data/NHANES.npy --pairs "0,1;2,3" --K 10

For library usage::

    from code import crud_test, crud_z_test
"""

# Re-export the public API from the inner package for backward compatibility.
from crud.crud_test import (  # noqa: F401
    crud_test,
    crud_z_test,
    CrudTestResult,
    CrudZResult,
    main,
)

if __name__ == "__main__":
    main()
