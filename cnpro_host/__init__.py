"""L2 -- the host-bound half of CNPro.

Everything here may import the host freely (backend.*, modules_forge.*).
Nothing here may be imported BY cnpro_core: the dependency arrow points one way
only, which is what keeps cnpro_core portable. See ARCHITECTURE.md.
"""
