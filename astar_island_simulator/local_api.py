"""Drop-in local API replacement for offline testing.

Mimics the real API endpoints so you can test main.py without touching
the live server. Uses AstarIslandSimulator internally.

Usage:
    from astar_island_simulator.local_api import LocalAPI

    api = LocalAPI(n_seeds=5, map_width=40, map_height=40,
                   queries_max=50, base_map_seed=42)
    # Use api.session in place of your real requests.Session
"""
from __future__ import annotations

import uuid
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .env import (AstarIslandSimulator, HiddenParams, Settlement,
                        CODE_TO_CLASS, NUM_CLASSES)


class LocalAPI:
    """Simulates the Astar Island REST API locally.

    After construction, call the methods directly or use ``get_session()``
    to get a mock session object whose ``.get()`` / ``.post()`` calls
    route to the local simulator.
    """

    def __init__(
        self,
        n_seeds: int = 5,
        map_width: int = 40,
        map_height: int = 40,
        queries_max: int = 50,
        base_map_seed: int = 42,
        params: Optional[HiddenParams] = None,
    ):
        self.round_id = str(uuid.uuid4())
        self.round_number = 1
        self.n_seeds = n_seeds
        self.map_width = map_width
        self.map_height = map_height
        self.queries_max = queries_max
        self.queries_used = 0
        self.params = params or HiddenParams()
        self._next_sim_seed = 1000

        # Build a simulator per seed (different map_seed)
        self.simulators: list[AstarIslandSimulator] = []
        for i in range(n_seeds):
            sim = AstarIslandSimulator(
                map_seed=base_map_seed + i,
                params=self.params,
                width=map_width, height=map_height,
            )
            self.simulators.append(sim)

    # ── Endpoint implementations ──────────────────────────────────────

    def get_rounds(self) -> list[dict]:
        return [{
            "id": self.round_id,
            "round_number": self.round_number,
            "status": "active",
            "map_width": self.map_width,
            "map_height": self.map_height,
        }]

    def get_round_detail(self, round_id: str) -> dict:
        initial_states = []
        for sim in self.simulators:
            grid, setts = sim.initial_state()
            initial_states.append({"grid": grid, "settlements": setts})
        return {
            "id": self.round_id,
            "round_number": self.round_number,
            "status": "active",
            "map_width": self.map_width,
            "map_height": self.map_height,
            "seeds_count": self.n_seeds,
            "initial_states": initial_states,
        }

    def get_budget(self) -> dict:
        return {
            "round_id": self.round_id,
            "queries_used": self.queries_used,
            "queries_max": self.queries_max,
            "active": True,
        }

    def simulate(self, round_id: str, seed_index: int,
                 viewport_x: int = 0, viewport_y: int = 0,
                 viewport_w: int = 15, viewport_h: int = 15) -> dict:
        if self.queries_used >= self.queries_max:
            return {"error": "Budget exhausted"}

        sim = self.simulators[seed_index]
        sim_seed = self._next_sim_seed
        self._next_sim_seed += 1
        self.queries_used += 1

        final_grid, final_settlements = sim.run(sim_seed)

        # Clamp viewport
        vx = max(0, min(viewport_x, self.map_width - 5))
        vy = max(0, min(viewport_y, self.map_height - 5))
        vw = max(5, min(viewport_w, self.map_width - vx))
        vh = max(5, min(viewport_h, self.map_height - vy))

        # Extract viewport grid
        vp_grid = final_grid[vy:vy + vh, vx:vx + vw].tolist()

        # Extract settlements in viewport
        vp_setts = []
        for s in final_settlements:
            if vx <= s.x < vx + vw and vy <= s.y < vy + vh:
                vp_setts.append(s.to_full_dict())

        return {
            "grid": vp_grid,
            "settlements": vp_setts,
            "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
            "width": self.map_width,
            "height": self.map_height,
            "queries_used": self.queries_used,
            "queries_max": self.queries_max,
        }

    def submit(self, round_id: str, seed_index: int,
               prediction: list) -> dict:
        return {
            "status": "accepted",
            "round_id": round_id,
            "seed_index": seed_index,
        }

    def score_prediction(self, seed_index: int, prediction: np.ndarray,
                         n_sims: int = 200) -> dict:
        """Score a prediction against simulated ground truth.

        Returns entropy-weighted KL divergence per cell and total score.
        This is the key method for offline evaluation.
        """
        sim = self.simulators[seed_index]
        gt = sim.ground_truth_distribution(n_sims=n_sims)

        H, W = self.map_height, self.map_width
        kl_per_cell = np.zeros((H, W))
        entropy_per_cell = np.zeros((H, W))
        eps = 1e-10

        for y in range(H):
            for x in range(W):
                q = gt[y, x]  # ground truth
                p = prediction[y, x]  # our prediction

                # Entropy of ground truth
                h = -np.sum(q * np.log(q + eps))
                entropy_per_cell[y, x] = h

                # KL(q || p)
                kl = np.sum(q * np.log((q + eps) / (p + eps)))
                kl_per_cell[y, x] = kl

        # Entropy-weighted KL
        total_entropy = entropy_per_cell.sum()
        if total_entropy > 0:
            weights = entropy_per_cell / total_entropy
        else:
            weights = np.ones((H, W)) / (H * W)

        weighted_kl = (kl_per_cell * weights).sum()

        # Score: higher is better. Base 100, subtract weighted KL penalty
        # (approximate scoring; real scoring formula may differ)
        score = max(0, 100 + 20 * (1 - weighted_kl))

        return {
            "score": round(score, 4),
            "weighted_kl": round(float(weighted_kl), 6),
            "mean_kl": round(float(kl_per_cell.mean()), 6),
            "mean_entropy": round(float(entropy_per_cell.mean()), 6),
            "n_dynamic_cells": int((entropy_per_cell > 0.01).sum()),
            "kl_per_cell": kl_per_cell,
            "entropy_per_cell": entropy_per_cell,
        }

    # ── Mock session ──────────────────────────────────────────────────

    def get_session(self) -> "MockSession":
        """Return a mock requests.Session that routes to this local API."""
        return MockSession(self)


class _MockResponse:
    """Mimics requests.Response."""

    def __init__(self, data: Any, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = json.dumps(data, default=str)[:500]

    def json(self):
        return self._data


class MockSession:
    """Drop-in replacement for requests.Session that routes to LocalAPI."""

    def __init__(self, api: LocalAPI):
        self.api = api
        self.cookies = _MockCookies()
        self.headers = {}

    def get(self, url: str, **kwargs) -> _MockResponse:
        path = url.split("/astar-island")[-1] if "/astar-island" in url else url
        path = path.rstrip("/")

        if path == "/rounds":
            return _MockResponse(self.api.get_rounds())
        elif path.startswith("/rounds/") and "/" not in path[len("/rounds/"):]:
            round_id = path.split("/")[-1]
            return _MockResponse(self.api.get_round_detail(round_id))
        elif path == "/budget":
            return _MockResponse(self.api.get_budget())
        elif "/analysis/" in path:
            return _MockResponse({"detail": "Not available in local mode"},
                                 status_code=404)
        elif path == "/leaderboard":
            return _MockResponse([])
        elif path == "/my-rounds":
            return _MockResponse([])
        return _MockResponse({"error": f"Unknown endpoint: {url}"}, 404)

    def post(self, url: str, json: dict = None, **kwargs) -> _MockResponse:
        if url.endswith("/simulate"):
            data = json or {}
            result = self.api.simulate(
                round_id=data.get("round_id", ""),
                seed_index=data.get("seed_index", 0),
                viewport_x=data.get("viewport_x", 0),
                viewport_y=data.get("viewport_y", 0),
                viewport_w=data.get("viewport_w", 15),
                viewport_h=data.get("viewport_h", 15),
            )
            return _MockResponse(result)
        elif url.endswith("/submit"):
            data = json or {}
            result = self.api.submit(
                round_id=data.get("round_id", ""),
                seed_index=data.get("seed_index", 0),
                prediction=data.get("prediction", []),
            )
            return _MockResponse(result)
        return _MockResponse({"error": f"Unknown endpoint: {url}"}, 404)


class _MockCookies:
    def set(self, name, value):
        pass
