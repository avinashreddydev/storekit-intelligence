import numpy as np
from typing import Callable, Optional

class PoissonDemand:
    """
    Poisson demand with calendar-based seasonality:
        D_t ~ Poisson(lambda_t),  lambda_t = lambda_base * (1 + alpha * f(t))

    Parameters
    ----------
    lambda_base : float
        Baseline demand rate (items/day).
    alpha : float, optional (default: 0.0)
        Amplitude of seasonal variation. Use 0.0 for no seasonality.
        Typical range: [0, 1]. Values outside this range are allowed but be mindful of negativity after modulation.
    f : Callable[[int], float], optional
        Seasonal function f(t) -> ℝ. Interpreted as a (preferably bounded) periodic signal
        encoding calendar effects (e.g., holidays). If None, a sinusoid of period `period`
        is used: f(t) = sin(2π * (t % period) / period).
    period : int, optional (default: 365)
        Period for the default sinusoid f(t) when `f` is None.
    rng : np.random.Generator, optional
        Random number generator. If None, a default_rng() is created.

    Notes
    -----
    - We enforce lambda_t >= 1e-8 to avoid numerical issues.
    - If you provide a custom f(t), you may want to keep it roughly in [-1, 1].
    """

    def __init__(
        self,
        lambda_base: float,
        alpha: float = 0.0,
        f: Optional[Callable[[int], float]] = None,
        period: int = 365,
        rng: Optional[np.random.Generator] = None,
    ):
        self.lambda_base = float(lambda_base)
        self.alpha = float(alpha)
        self.period = int(period)
        self.f = f if f is not None else self._default_f
        self.rng = rng or np.random.default_rng()

    # Default seasonal function: smooth sinusoid over the specified period
    def _default_f(self, t: int) -> float:
        return np.sin(2.0 * np.pi * ((t % self.period) / self.period))

    def lambda_t(self, day_idx: int) -> float:
        """Compute λ_t = λ_base * (1 + α * f(t)), clipped to be nonnegative."""
        lam = self.lambda_base * (1.0 + self.alpha * float(self.f(day_idx)))
        return max(1e-8, lam)

    def sample(self, days: int, start_day: int) -> np.ndarray:
        """
        Draw integer demands for days: start_day, ..., start_day + days - 1.

        Returns
        -------
        np.ndarray of shape (days,), dtype=int
        """
        out = np.zeros(days, dtype=np.int32)
        for d in range(days):
            lam = self.lambda_t(start_day + d)
            out[d] = self.rng.poisson(lam)
        return out
