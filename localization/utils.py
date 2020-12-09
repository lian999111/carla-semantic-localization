import minisam as ms

def copy_se2(se2):
    """Deep copy SE2 object.

    Args:
        se2: SE2 object.
    """
    trans = se2.translation()
    so2 = se2.so2()
    return ms.sophus.SE2(so2, trans)