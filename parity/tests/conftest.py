"""Hypothesis profiles. Fixtures and --impl live in pyroid_parity.plugin (loaded via addopts)."""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("dev", max_examples=200, deadline=None)
settings.register_profile("ci", max_examples=2000, deadline=None, print_blob=True,
                          suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
