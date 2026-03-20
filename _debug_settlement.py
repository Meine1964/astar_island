"""Trace settlement economy to debug why they all die, and test alternative formulas."""
import math
import numpy as np
from astar_island_simulator.env import HiddenParams

p = HiddenParams()
print("=== Current HiddenParams ===")
print(f"  winter_base_severity={p.winter_base_severity}, variance={p.winter_severity_variance}")
print(f"  collapse_food_threshold={p.collapse_food_threshold}")
print(f"  food_per_forest={p.food_per_forest}, food_per_plains={p.food_per_plains}")
print(f"  growth_food_threshold={p.growth_food_threshold}")
print(f"  expansion_pop_threshold={p.expansion_pop_threshold}")

# Typical food gains for different terrain mixes
scenarios = [
    ("Rich (4F+3P)", 4 * p.food_per_forest + 3 * p.food_per_plains),
    ("Average (2F+4P)", 2 * p.food_per_forest + 4 * p.food_per_plains),
    ("Poor (1F+5P)", 1 * p.food_per_forest + 5 * p.food_per_plains),
]

# Test different food_loss formulas
formulas = {
    "CURRENT (sqrt)":   lambda sev, pop: sev * (1 + 0.3 * (pop ** 0.5)),
    "Linear 0.02":      lambda sev, pop: sev * (1.0 + 0.02 * pop),
    "Log":              lambda sev, pop: sev * (1.0 + 0.3 * math.log(1 + pop)),
    "Flat 1.5":         lambda sev, pop: sev * 1.5,
}

rng = np.random.default_rng(42)

for fname, food_loss_fn in formulas.items():
    print(f"\n{'='*60}")
    print(f"Formula: {fname}")
    print(f"{'='*60}")

    for sname, food_gain in scenarios:
        # Run 100 trials
        n_trials = 200
        survived = 0
        final_pops = []
        death_years = []

        for trial in range(n_trials):
            pop = 10
            food = 5.0
            alive = True
            for year in range(50):
                food += food_gain
                # Growth
                if food > p.growth_food_threshold:
                    growth = max(1, int(food / p.growth_food_threshold))
                    pop += growth
                    food -= growth * 1.5  # original cost
                # Winter
                sev = max(0, p.winter_base_severity + rng.normal(0, p.winter_severity_variance))
                loss = food_loss_fn(sev, pop)
                food -= loss
                if sev > 5:
                    pop = max(1, pop - max(1, int(pop * 0.1)))
                if food < p.collapse_food_threshold or pop <= 0:
                    alive = False
                    death_years.append(year)
                    break
            if alive:
                survived += 1
                final_pops.append(pop)

        avg_death = np.mean(death_years) if death_years else float("nan")
        avg_pop = np.mean(final_pops) if final_pops else 0
        print(f"  {sname:20s}: survive={survived}/{n_trials} ({100*survived/n_trials:.0f}%) "
              f"avg_final_pop={avg_pop:.0f}  avg_death_year={avg_death:.1f}")
