"""
hxtorch.so will load this module and execute patch() during initialization to
patch the python part of hxtorch.
"""

def patch(module):
    """
    This hook will be executed at the end of hxtorch module generation.
    """
    from pyhxtorch import nn
    module.nn = nn
