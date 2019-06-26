
def line_search(model, param0, direction, step_size, num_steps, args=None):
    """
    Parameters
    ----------
    model : callable(param0, *args)->float
        The objective function to call
    param0 :
        Initial parameter vector
    direction :
        Direction to search in
    step_size :
        Size of steps in search
    num_steps :
        Number of steps in line search

    Returns
    -------
    List of objective function values at steps
    """
    out = []
    for step in range(num_steps):
        out.append(model(param0 + step*step_size*direction))

    return out
