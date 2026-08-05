"""Audit package: runner, pricing, report.

This package is itself callable: density.audit(path, ...) runs the full
audit pipeline. The spec requires both a density.audit(...) function on
the SDK surface and a density/audit/ package holding runner, pricing,
and report. Python cannot give one name to both: importing any submodule
(density.audit.pricing, say) rebinds the density.audit attribute from
the function to this module object, silently shadowing the function.
Making the module object callable resolves the collision for good: both
import styles work no matter the import order.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class _CallableAuditModule(types.ModuleType):
    """Module type whose call forwards to audit.runner.run_audit."""

    def __call__(self, path: Any, out: str = "report.html", **kwargs: Any) -> Any:
        # Imported lazily so `import density` stays light: the audit
        # stack pulls in jinja2 and the whole engine.
        from density.audit.runner import run_audit

        return run_audit(path, out=out, **kwargs)


sys.modules[__name__].__class__ = _CallableAuditModule
