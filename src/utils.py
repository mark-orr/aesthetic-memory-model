# Math helpers and data transformations go here

def inverted_parabola(x, r):
    """f(x, r) = 1 - (x/r)^2"""
    return 1 - (x / r) ** 2


def inverted_parabola_left_anchored(x, r):
    """Inverted parabola with left x-intercept fixed at 0, vertex y=1 (at
    x=r/2), and right x-intercept at x=r -- so r parameterizes the full
    width of the parabola on the x-axis."""
    return 1 - (2 * x / r - 1) ** 2
