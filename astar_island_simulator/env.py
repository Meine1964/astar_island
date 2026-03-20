"""Astar Island simulation engine.

Approximate reimplementation of the server-side Norse civilisation simulator.
The exact hidden parameters are unknown; this uses reasonable defaults that
can be calibrated against ground truth from completed rounds.

Usage:
    sim = AstarIslandSimulator(map_seed=42, params=HiddenParams())
    grid, settlements = sim.initial_state()
    final_grid, final_settlements = sim.run(sim_seed=123)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Terrain codes (matching the real API)
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

# Prediction class mapping
CODE_TO_CLASS = {OCEAN: 0, PLAINS: 0, EMPTY: 0,
                 SETTLEMENT: 1, PORT: 2, RUIN: 3, FOREST: 4, MOUNTAIN: 5}
NUM_CLASSES = 6

NEIGHBOURS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBOURS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]


# ── Hidden parameters ────────────────────────────────────────────────

@dataclass
class HiddenParams:
    """Tunable parameters that control the simulation behaviour.

    These correspond to the *hidden parameters* that the real server uses.
    All seeds in a round share the same hidden params.
    """
    # -- Map generation --
    n_fjords: int = 4
    fjord_max_length: int = 12
    n_mountain_chains: int = 3
    mountain_chain_length: int = 10
    n_forest_patches: int = 8
    forest_patch_radius: int = 4
    n_initial_settlements: int = 12
    min_settlement_spacing: int = 4

    # -- Growth --
    food_per_forest: float = 3.0        # food from each adjacent forest cell
    food_per_plains: float = 1.0        # food from each adjacent plains cell
    growth_food_threshold: float = 8.0  # food needed to grow population
    port_wealth_threshold: float = 6.0  # wealth needed to become a port
    longship_tech_threshold: float = 5.0
    expansion_pop_threshold: int = 15   # population to found new settlement
    expansion_radius: int = 3

    # -- Conflict --
    raid_range: int = 4
    longship_raid_bonus: int = 6        # extra range with longship
    desperate_food_threshold: float = 3.0
    raid_strength_factor: float = 0.3
    raid_loot_fraction: float = 0.2
    conquest_threshold: float = 0.8     # damage ratio to trigger allegiance flip

    # -- Trade --
    trade_range: int = 6
    trade_food_bonus: float = 2.0
    trade_wealth_bonus: float = 2.0
    tech_diffusion_rate: float = 0.1

    # -- Winter --
    winter_base_severity: float = 2.0
    winter_severity_variance: float = 1.5
    collapse_food_threshold: float = -3.0  # below this -> ruin

    # -- Environment --
    reclaim_radius: int = 3
    reclaim_pop_threshold: int = 8
    forest_regrowth_prob: float = 0.15    # prob ruin becomes forest per year
    plains_regrowth_prob: float = 0.10    # prob ruin becomes plains per year

    # -- Simulation --
    n_years: int = 50


# ── Settlement ────────────────────────────────────────────────────────

@dataclass
class Settlement:
    x: int
    y: int
    population: int = 10
    food: float = 5.0
    wealth: float = 1.0
    defense: float = 3.0
    tech_level: float = 1.0
    has_port: bool = False
    has_longship: bool = False
    owner_id: int = 0
    alive: bool = True
    damage_taken: float = 0.0  # accumulated raid damage this year

    def to_initial_dict(self):
        """Public info visible in round details."""
        return {"x": self.x, "y": self.y,
                "has_port": self.has_port, "alive": self.alive}

    def to_full_dict(self):
        """Full info visible through simulate queries."""
        return {"x": self.x, "y": self.y,
                "population": self.population,
                "food": round(self.food, 2),
                "wealth": round(self.wealth, 2),
                "defense": round(self.defense, 2),
                "has_port": self.has_port, "alive": self.alive,
                "owner_id": self.owner_id}


# ── Map generation ────────────────────────────────────────────────────

def generate_map(width: int, height: int, map_seed: int,
                 params: HiddenParams) -> tuple[np.ndarray, list[Settlement]]:
    """Procedurally generate terrain and initial settlements."""
    rng = np.random.default_rng(map_seed)
    grid = np.full((height, width), PLAINS, dtype=int)

    # Ocean borders
    grid[0, :] = OCEAN
    grid[height - 1, :] = OCEAN
    grid[:, 0] = OCEAN
    grid[:, width - 1] = OCEAN

    # Fjords: cut inland from random border positions
    for _ in range(params.n_fjords):
        edge = rng.integers(0, 4)
        if edge == 0:     # top
            x, y, dx, dy = rng.integers(2, width - 2), 0, 0, 1
        elif edge == 1:   # bottom
            x, y, dx, dy = rng.integers(2, width - 2), height - 1, 0, -1
        elif edge == 2:   # left
            x, y, dx, dy = 0, rng.integers(2, height - 2), 1, 0
        else:             # right
            x, y, dx, dy = width - 1, rng.integers(2, height - 2), -1, 0

        length = rng.integers(3, params.fjord_max_length + 1)
        for step in range(length):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < width and 0 <= ny < height:
                grid[ny, nx] = OCEAN
                # Widen fjord by 1 cell perpendicular
                if dy == 0:  # horizontal fjord
                    for oy in [-1, 1]:
                        if 0 <= ny + oy < height:
                            grid[ny + oy, nx] = OCEAN
                else:        # vertical fjord
                    for ox in [-1, 1]:
                        if 0 <= nx + ox < width:
                            grid[ny, nx + ox] = OCEAN
            # Random walk drift
            if rng.random() < 0.3:
                if dy == 0:
                    dy += rng.choice([-1, 1])
                    dy = max(-1, min(1, dy))
                else:
                    dx += rng.choice([-1, 1])
                    dx = max(-1, min(1, dx))

    # Mountain chains via random walks
    for _ in range(params.n_mountain_chains):
        mx = rng.integers(3, width - 3)
        my = rng.integers(3, height - 3)
        for step in range(params.mountain_chain_length):
            if 1 <= mx < width - 1 and 1 <= my < height - 1:
                if grid[my, mx] != OCEAN:
                    grid[my, mx] = MOUNTAIN
            mx += rng.integers(-1, 2)
            my += rng.integers(-1, 2)
            mx = max(1, min(width - 2, mx))
            my = max(1, min(height - 2, my))

    # Forest patches (clustered groves)
    for _ in range(params.n_forest_patches):
        cx = rng.integers(2, width - 2)
        cy = rng.integers(2, height - 2)
        r = params.forest_patch_radius
        for fy in range(max(1, cy - r), min(height - 1, cy + r + 1)):
            for fx in range(max(1, cx - r), min(width - 1, cx + r + 1)):
                dist = abs(fx - cx) + abs(fy - cy)
                if dist <= r and grid[fy, fx] == PLAINS:
                    if rng.random() < 0.7 - 0.1 * dist:
                        grid[fy, fx] = FOREST

    # Place initial settlements on land cells, spaced apart
    settlements = []
    land_cells = [(x, y) for y in range(1, height - 1)
                  for x in range(1, width - 1)
                  if grid[y, x] in (PLAINS, EMPTY)]
    rng.shuffle(land_cells)

    faction_id = 0
    for (sx, sy) in land_cells:
        if len(settlements) >= params.n_initial_settlements:
            break
        # Check spacing
        too_close = False
        for s in settlements:
            if abs(s.x - sx) + abs(s.y - sy) < params.min_settlement_spacing:
                too_close = True
                break
        if too_close:
            continue

        # Check if coastal (adjacent to ocean) -> port candidate
        coastal = any(
            0 <= sx + dx < width and 0 <= sy + dy < height
            and grid[sy + dy, sx + dx] == OCEAN
            for dx, dy in NEIGHBOURS_4
        )

        s = Settlement(
            x=sx, y=sy,
            population=rng.integers(6, 16),
            food=float(rng.integers(3, 8)),
            wealth=float(rng.integers(1, 4)),
            defense=float(rng.integers(2, 6)),
            tech_level=float(rng.integers(1, 3)),
            has_port=coastal and rng.random() < 0.5,
            owner_id=faction_id,
        )
        grid[sy, sx] = PORT if s.has_port else SETTLEMENT
        settlements.append(s)
        faction_id += 1

    return grid, settlements


# ── Simulation engine ─────────────────────────────────────────────────

class AstarIslandSimulator:
    """Run the Norse civilisation simulation."""

    def __init__(self, map_seed: int, params: Optional[HiddenParams] = None,
                 width: int = 40, height: int = 40):
        self.map_seed = map_seed
        self.params = params or HiddenParams()
        self.width = width
        self.height = height

        self.base_grid, self.base_settlements = generate_map(
            width, height, map_seed, self.params)

    def initial_state(self) -> tuple[list[list[int]], list[dict]]:
        """Return the initial grid and settlement list (public info only)."""
        grid = self.base_grid.tolist()
        setts = [s.to_initial_dict() for s in self.base_settlements]
        return grid, setts

    def run(self, sim_seed: int) -> tuple[np.ndarray, list[Settlement]]:
        """Run a full 50-year simulation with the given random seed.

        Returns the final grid and settlement list.
        """
        rng = np.random.default_rng(sim_seed)
        p = self.params
        grid = self.base_grid.copy()
        # Deep copy settlements
        settlements = [Settlement(
            x=s.x, y=s.y, population=s.population, food=s.food,
            wealth=s.wealth, defense=s.defense, tech_level=s.tech_level,
            has_port=s.has_port, has_longship=s.has_longship,
            owner_id=s.owner_id, alive=s.alive
        ) for s in self.base_settlements]

        for year in range(p.n_years):
            settlements = self._phase_growth(grid, settlements, rng)
            settlements = self._phase_conflict(grid, settlements, rng)
            settlements = self._phase_trade(grid, settlements, rng)
            settlements = self._phase_winter(grid, settlements, rng)
            settlements = self._phase_environment(grid, settlements, rng)

        return grid, settlements

    def run_grid(self, sim_seed: int) -> np.ndarray:
        """Run simulation and return only the final grid."""
        grid, _ = self.run(sim_seed)
        return grid

    def ground_truth_distribution(self, n_sims: int = 200,
                                  base_seed: int = 0) -> np.ndarray:
        """Run many simulations and compute per-cell class distributions.

        Returns an (H, W, 6) array of probabilities (ground truth).
        """
        counts = np.zeros((self.height, self.width, NUM_CLASSES))
        for i in range(n_sims):
            grid = self.run_grid(base_seed + i)
            for y in range(self.height):
                for x in range(self.width):
                    cls = CODE_TO_CLASS.get(int(grid[y, x]), 0)
                    counts[y, x, cls] += 1
        return counts / n_sims

    # ── Phase implementations ─────────────────────────────────────────

    def _alive(self, settlements):
        return [s for s in settlements if s.alive]

    def _adjacent_terrain_counts(self, grid, x, y):
        """Count terrain types in the 8 neighbours."""
        counts = {}
        for dx, dy in NEIGHBOURS_8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                t = int(grid[ny, nx])
                counts[t] = counts.get(t, 0) + 1
        return counts

    def _is_coastal(self, grid, x, y):
        for dx, dy in NEIGHBOURS_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if grid[ny, nx] == OCEAN:
                    return True
        return False

    def _distance(self, s1, s2):
        return abs(s1.x - s2.x) + abs(s1.y - s2.y)

    # -- Growth phase --

    def _phase_growth(self, grid, settlements, rng):
        p = self.params
        for s in self._alive(settlements):
            adj = self._adjacent_terrain_counts(grid, s.x, s.y)

            # Food production
            food_gain = (adj.get(FOREST, 0) * p.food_per_forest +
                         adj.get(PLAINS, 0) * p.food_per_plains +
                         adj.get(EMPTY, 0) * p.food_per_plains * 0.5)
            s.food += food_gain

            # Population growth
            if s.food > p.growth_food_threshold:
                growth = max(1, int(s.food / p.growth_food_threshold))
                s.population += growth
                s.food -= growth * 1.5

            # Tech growth (slow)
            s.tech_level += 0.05 + 0.01 * s.wealth

            # Defense grows with population
            s.defense = max(s.defense, s.population * 0.3)

            # Wealth from population
            s.wealth += s.population * 0.05

            # Port development
            if not s.has_port and self._is_coastal(grid, s.x, s.y):
                if s.wealth >= p.port_wealth_threshold and rng.random() < 0.3:
                    s.has_port = True
                    grid[s.y, s.x] = PORT

            # Longship building
            if s.has_port and not s.has_longship:
                if s.tech_level >= p.longship_tech_threshold and rng.random() < 0.2:
                    s.has_longship = True

            # Expansion: found new settlement
            if s.population >= p.expansion_pop_threshold and rng.random() < 0.15:
                candidates = []
                r = p.expansion_radius
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = s.x + dx, s.y + dy
                        if (0 < nx < self.width - 1 and 0 < ny < self.height - 1
                                and grid[ny, nx] in (PLAINS, EMPTY)):
                            # Not too close to another settlement
                            ok = all(abs(o.x - nx) + abs(o.y - ny) >= 2
                                     for o in self._alive(settlements))
                            if ok:
                                candidates.append((nx, ny))
                if candidates:
                    nx, ny = candidates[rng.integers(len(candidates))]
                    coastal = self._is_coastal(grid, nx, ny)
                    new_s = Settlement(
                        x=nx, y=ny,
                        population=max(3, s.population // 4),
                        food=s.food * 0.3,
                        wealth=s.wealth * 0.2,
                        defense=2.0,
                        tech_level=s.tech_level * 0.8,
                        has_port=coastal and rng.random() < 0.3,
                        owner_id=s.owner_id,
                    )
                    grid[ny, nx] = PORT if new_s.has_port else SETTLEMENT
                    settlements.append(new_s)
                    s.population -= new_s.population
                    s.food -= new_s.food

        return settlements

    # -- Conflict phase --

    def _phase_conflict(self, grid, settlements, rng):
        p = self.params
        alive = self._alive(settlements)

        for attacker in alive:
            if not attacker.alive:
                continue

            # Desperate settlements raid more often
            desperate = attacker.food < p.desperate_food_threshold
            raid_prob = 0.4 if desperate else 0.15

            if rng.random() > raid_prob:
                continue

            # Find targets in range
            max_range = p.raid_range + (p.longship_raid_bonus if attacker.has_longship else 0)
            targets = [t for t in alive if t.alive and t is not attacker
                       and t.owner_id != attacker.owner_id
                       and self._distance(attacker, t) <= max_range]
            if not targets:
                continue

            target = targets[rng.integers(len(targets))]
            attack_str = (attacker.population * p.raid_strength_factor
                          + attacker.defense * 0.3
                          + attacker.tech_level * 0.2)
            defend_str = (target.defense + target.population * 0.2)

            if attack_str > defend_str * (0.6 if desperate else 0.8):
                # Successful raid
                loot_food = target.food * p.raid_loot_fraction
                loot_wealth = target.wealth * p.raid_loot_fraction
                attacker.food += loot_food
                attacker.wealth += loot_wealth
                target.food -= loot_food
                target.wealth -= loot_wealth
                target.damage_taken += attack_str * 0.3
                target.defense = max(0, target.defense - attack_str * 0.1)

                # Conquest check
                if target.damage_taken > target.population * p.conquest_threshold:
                    target.owner_id = attacker.owner_id
                    target.damage_taken = 0

        # Reset damage for next year
        for s in alive:
            s.damage_taken = 0

        return settlements

    # -- Trade phase --

    def _phase_trade(self, grid, settlements, rng):
        p = self.params
        ports = [s for s in self._alive(settlements) if s.has_port]

        for i, port_a in enumerate(ports):
            for port_b in ports[i + 1:]:
                if port_a.owner_id == port_b.owner_id:
                    at_war = False
                else:
                    # Simple war check: recent conflict between factions
                    at_war = rng.random() < 0.3

                if at_war:
                    continue
                if self._distance(port_a, port_b) > p.trade_range:
                    continue

                # Trade happens
                port_a.food += p.trade_food_bonus
                port_b.food += p.trade_food_bonus
                port_a.wealth += p.trade_wealth_bonus
                port_b.wealth += p.trade_wealth_bonus

                # Tech diffusion
                avg_tech = (port_a.tech_level + port_b.tech_level) / 2
                port_a.tech_level += (avg_tech - port_a.tech_level) * p.tech_diffusion_rate
                port_b.tech_level += (avg_tech - port_b.tech_level) * p.tech_diffusion_rate

        return settlements

    # -- Winter phase --

    def _phase_winter(self, grid, settlements, rng):
        p = self.params
        severity = max(0, p.winter_base_severity
                       + rng.normal(0, p.winter_severity_variance))

        for s in self._alive(settlements):
            # Food loss
            food_loss = severity * (1 + s.population * 0.1)
            s.food -= food_loss

            # Population loss from harsh winter
            if severity > 4:
                pop_loss = max(1, int(severity * 0.3))
                s.population = max(0, s.population - pop_loss)

            # Collapse check
            if s.food < p.collapse_food_threshold or s.population <= 0:
                s.alive = False
                grid[s.y, s.x] = RUIN

                # Disperse population to nearby friendly settlements
                nearby = [o for o in self._alive(settlements)
                          if o.owner_id == s.owner_id
                          and self._distance(s, o) <= 5]
                if nearby and s.population > 0:
                    refugees_each = max(1, s.population // len(nearby))
                    for o in nearby:
                        o.population += refugees_each
                s.population = 0

        return settlements

    # -- Environment phase --

    def _phase_environment(self, grid, settlements, rng):
        p = self.params
        alive = self._alive(settlements)

        # Find all ruins on the map
        ruins = []
        for y in range(self.height):
            for x in range(self.width):
                if grid[y, x] == RUIN:
                    ruins.append((x, y))

        for rx, ry in ruins:
            # Check if a nearby thriving settlement reclaims it
            reclaimed = False
            for s in alive:
                if self._distance(s, Settlement(x=rx, y=ry)) <= p.reclaim_radius:
                    if s.population >= p.reclaim_pop_threshold and rng.random() < 0.2:
                        coastal = self._is_coastal(grid, rx, ry)
                        new_s = Settlement(
                            x=rx, y=ry,
                            population=max(2, s.population // 5),
                            food=s.food * 0.2,
                            wealth=s.wealth * 0.15,
                            defense=1.5,
                            tech_level=s.tech_level * 0.7,
                            has_port=coastal and rng.random() < 0.4,
                            owner_id=s.owner_id,
                        )
                        grid[ry, rx] = PORT if new_s.has_port else SETTLEMENT
                        settlements.append(new_s)
                        s.population -= new_s.population
                        reclaimed = True
                        break

            if not reclaimed:
                # Natural reclamation
                if rng.random() < p.forest_regrowth_prob:
                    grid[ry, rx] = FOREST
                elif rng.random() < p.plains_regrowth_prob:
                    grid[ry, rx] = PLAINS

        return settlements
