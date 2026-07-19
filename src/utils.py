# Math helpers and data transformations go here

def inverted_parabola(x, r):
    """f(x, r) = 1 - (x/r)^2"""
    return 1 - (x / r) ** 2


def inverted_parabola_left_anchored(x, r):
    """Inverted parabola with left x-intercept fixed at 0, vertex y=1 (at
    x=r/2), and right x-intercept at x=r -- so r parameterizes the full
    width of the parabola on the x-axis."""
    return 1 - (2 * x / r - 1) ** 2


def compute_convergence_metrics(df, threshold=0.9, sustain_window=50, steady_state_window=200):
    """convergence_trial: the `trial` value at the first index where
    `evaluation` stays >= threshold for `sustain_window` consecutive rows
    (None if that never happens). steady_state_mean: mean `evaluation` over
    `steady_state_window` rows starting at that point (None if convergence
    was never reached)."""
    evaluations = df["evaluation"].values
    convergence_idx = None
    for i in range(len(evaluations) - sustain_window + 1):
        if (evaluations[i:i + sustain_window] >= threshold).all():
            convergence_idx = i
            break

    if convergence_idx is None:
        return {"convergence_trial": None, "steady_state_mean": None}

    convergence_trial = df["trial"].iloc[convergence_idx]
    steady_state_mean = evaluations[convergence_idx:convergence_idx + steady_state_window].mean()
    return {"convergence_trial": convergence_trial, "steady_state_mean": steady_state_mean}
