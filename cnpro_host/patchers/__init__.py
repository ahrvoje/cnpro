"""Control-model patchers: one module per injection mechanism.

Each owns its injection path (`*_impl.py`) and declares what it can express
(`supports_*` on CNProModelPatcher). Adding a model family = adding a module
here plus one line in ../registry.py.
"""
