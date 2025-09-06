import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from demand import PoissonDemand

# ---------- Store Environment ----------
class StoreEnv(gym.Env):
    """
    Single-product store.
    Action = number of units to top-up at the beginning of each step.
    Each step simulates 'step_days' consecutive days of demand fulfillment.

    Reward per step = -(unmet_demand + holding_cost),
    where holding cost is stock_rent_per_item_per_day * (inventory at end of each day),
    accumulated over the simulated 'step_days' days.

    Observation (Dict):
      - day: scalar (current day index)
      - stock: scalar (current on-hand stock before top-up)
      - recent_sales: length-H vector of last H fulfilled daily sales
      - season: {0,1} indicator for the *next* day block (simple season flag)
      - lam_hint: float hint of current demand rate (optional, helps learning)

    Episode ends after total_days are simulated.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        demand,                            # object with .sample(days, start_day, season_flags) -> int array
        render_mode: str | None = None,
        total_days: int = 365,
        step_days: int = 5,
        init_stock: int = 0,
        stock_rent_per_item_per_day: float = 0.05,
        max_topup: int = 500,              # cap for discrete action
        history_len: int = 14,
        season_func=None,                  # optional: f(day_idx)->{0,1}; if None, uses simple sinusoid threshold
        lam_hint: bool = True,             # include demand rate hint in observation
        window_size: int = 512,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Core config
        self.demand = demand
        self.total_days = int(total_days)
        self.step_days = int(step_days)
        self.init_stock = int(init_stock)
        self.stock_rent = float(stock_rent_per_item_per_day)
        self.max_topup = int(max_topup)
        self.history_len = int(history_len)
        self.lam_hint_enabled = bool(lam_hint)

        # Simple season function if none provided (sinusoid > 0 => season)
        self.season_func = season_func or (lambda d: int(np.sin(2*np.pi*(d % 365)/365.0) > 0.0))

        # Action space: top-up amount (Discrete)
        self.action_space = spaces.Discrete(self.max_topup + 1)

        # Observation space
        # Use float32 boxes for simplicity and stable scaling.
        obs_dict = {
            "day": spaces.Box(low=0.0, high=float(self.total_days), shape=(1,), dtype=np.float32),
            "stock": spaces.Box(low=0.0, high=1e7, shape=(1,), dtype=np.float32),
            "recent_sales": spaces.Box(low=0.0, high=1e6, shape=(self.history_len,), dtype=np.float32),
            "season": spaces.Discrete(2),
        }
        if self.lam_hint_enabled:
            obs_dict["lam_hint"] = spaces.Box(low=0.0, high=1e6, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_dict)

        # Rendering / pygame
        self.window_size = int(window_size)
        self.window = None
        self.clock = None

        # State
        self._day = 0
        self._stock = 0
        self._recent_sales = np.zeros(self.history_len, dtype=np.float32)
        self._rng = np.random.default_rng()

    # ------------- Helpers -------------
    def _season_flags_for_block(self, start_day: int, days: int) -> np.ndarray:
        return np.array([self.season_func(start_day + d) for d in range(days)], dtype=np.int32)

    def _lam_hint_for_day(self, day_idx: int, season_flag: int) -> float:
        # Try to pull a rate hint from the demand object if it exposes it.
        # If not, we approximate by probing 1 day of demand mean via PoissonDemand._lam_for_day if present.
        lam = None
        if hasattr(self.demand, "_lam_for_day"):
            lam = float(self.demand._lam_for_day(day_idx, season_flag))
        # Fallback: no hint
        return 0.0 if lam is None else lam

    def _obs(self):
        season_now = int(self.season_func(self._day))
        obs = {
            "day": np.array([self._day], dtype=np.float32),
            "stock": np.array([self._stock], dtype=np.float32),
            "recent_sales": self._recent_sales.copy().astype(np.float32),
            "season": season_now,
        }
        if self.lam_hint_enabled:
            obs["lam_hint"] = np.array([self._lam_hint_for_day(self._day, season_now)], dtype=np.float32)
        return obs

    def _info(self, unmet, holding_cost, sales_block):
        return {
            "unmet_demand": float(unmet),
            "holding_cost": float(holding_cost),
            "sales_block_total": float(np.sum(sales_block)),
            "end_stock": int(self._stock),
            "start_day": int(self._day - self.step_days),
            "end_day": int(self._day - 1),
        }

    # ------------- Gym API -------------
    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            if hasattr(self.demand, "rng"):
                self.demand.rng = np.random.default_rng(seed + 1337)

        self._day = 0
        self._stock = int(self.init_stock)
        self._recent_sales = np.zeros(self.history_len, dtype=np.float32)

        if self.render_mode == "human":
            self._render_frame()
        return self._obs(), {}

    def step(self, action: int):
        # 1) Apply top-up
        action = int(action)
        action = max(0, min(self.max_topup, action))
        self._stock += action

        # 2) Simulate next block of days
        days = min(self.step_days, max(0, self.total_days - self._day))
        season_flags = self._season_flags_for_block(self._day, days)
        demand_block = self.demand.sample(days=days, start_day=self._day, season_flags=season_flags)

        sales_block = np.zeros(days, dtype=np.int32)
        unmet = 0
        holding_cost = 0.0

        for i in range(days):
            dem = int(demand_block[i])
            # fulfill
            sold = min(self._stock, dem)
            self._stock -= sold
            sales_block[i] = sold
            # unmet demand (penalty)
            unmet += (dem - sold)
            # holding cost for remaining stock at END of the day
            holding_cost += self.stock_rent * float(self._stock)

        # 3) Advance time
        self._day += days

        # 4) Update sales history (FIFO)
        if self.history_len > 0:
            take = min(self.history_len, days)
            if take < self.history_len:
                self._recent_sales[:-take] = self._recent_sales[take:]
                self._recent_sales[-take:] = sales_block[-take:]
            else:
                self._recent_sales[:] = sales_block[-self.history_len:]

        # 5) Reward
        reward = -(float(unmet) + float(holding_cost))

        # 6) Termination
        terminated = (self._day >= self.total_days)
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info(unmet, holding_cost, sales_block)

   


# ---------- Registration (adjust module path to your package layout) ----------
from gymnasium.envs.registration import register

register(
    id="gym_examples/StoreEnv-v0",
    entry_point="gym_examples.envs:StoreEnv",
    max_episode_steps=200,  # episode length in steps (each step = step_days)
)



if __name__ == "__main__":
    demand = PoissonDemand(lam_base=20.0, seasonal_amp=0.3, seasonal_period=365)
    env = StoreEnv(demand=demand, total_days=180, step_days=5, init_stock=100, stock_rent_per_item_per_day=0.02)

    obs, info = env.reset(seed=42)
    for _ in range(10):
        action = env.action_space.sample()  # replace with your policy's action
        obs, reward, term, trunc, info = env.step(action)
        # env.render()  # set render_mode="human" in ctor to visualize
        if term or trunc:
            break
    env.close()