"""Module 15 — reinforcement learning: MDPs, value iteration, Q-learning."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

__all__ = [
    "GridWorld",
    "value_iteration",
    "policy_iteration",
    "q_learning",
    "td_evaluate",
    "epsilon_greedy",
    "regret",
    "discounted_return",
]


class GridWorld:
    """A small stochastic grid world.

    Actions: 0 = up, 1 = down, 2 = left, 3 = right. With probability
    ``slip`` the agent takes a uniformly random action instead.
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, size: int = 5, goal=(0, 4), pit=(2, 2), walls=((1, 1), (3, 3)),
                 living: float = -0.04, gamma: float = 0.9, slip: float = 0.1,
                 goal_reward: float = 10.0, pit_reward: float = -10.0):
        self.n = size
        self.goal = tuple(goal)
        self.pit = tuple(pit)
        self.walls = set(map(tuple, walls))
        self.living = living
        self.gamma = gamma
        self.slip = slip
        self.goal_reward = goal_reward
        self.pit_reward = pit_reward
        self.start = (size - 1, 0)

    # ------------------------------------------------------------- dynamics
    def is_wall(self, r: int, c: int) -> bool:
        return not (0 <= r < self.n and 0 <= c < self.n) or (r, c) in self.walls

    def states(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.n) for c in range(self.n) if not self.is_wall(r, c)]

    def move(self, state, action: int):
        dr, dc = self.ACTIONS[action]
        nr, nc = state[0] + dr, state[1] + dc
        return state if self.is_wall(nr, nc) else (nr, nc)

    def step(self, state, action: int, rng: random.Random | None = None):
        rng = rng or random
        if rng.random() < self.slip:
            action = rng.randrange(4)
        nxt = self.move(state, action)
        reward = self.living
        if nxt == self.goal:
            reward += self.goal_reward
        if nxt == self.pit:
            reward += self.pit_reward
        return nxt, reward, nxt in (self.goal, self.pit)

    def transition_probs(self, state, action: int) -> dict[tuple[int, int], float]:
        probs: dict[tuple[int, int], float] = {}
        for a in range(4):
            p = (1 - self.slip) if a == action else self.slip / 3
            nxt = self.move(state, a)
            probs[nxt] = probs.get(nxt, 0.0) + p
        return probs

    def expected_reward(self, state, action: int) -> float:
        total = 0.0
        for nxt, p in self.transition_probs(state, action).items():
            r = self.living
            if nxt == self.goal:
                r += self.goal_reward
            if nxt == self.pit:
                r += self.pit_reward
            total += p * r
        return total


def value_iteration(env: GridWorld, tol: float = 1e-8, max_iters: int = 1000):
    """Bellman backups until convergence. Returns (V, policy, iterations)."""
    V = {s: 0.0 for s in env.states()}
    terminal = {env.goal, env.pit}
    for it in range(1, max_iters + 1):
        delta = 0.0
        new = dict(V)
        for s in env.states():
            if s in terminal:
                continue
            best = max(
                env.expected_reward(s, a)
                + env.gamma * sum(p * V[nxt] for nxt, p in env.transition_probs(s, a).items())
                for a in range(4)
            )
            delta = max(delta, abs(best - V[s]))
            new[s] = best
        V = new
        if delta < tol:
            break
    policy = {}
    for s in env.states():
        if s in terminal:
            continue
        policy[s] = max(
            range(4),
            key=lambda a: env.expected_reward(s, a)
            + env.gamma * sum(p * V[nxt] for nxt, p in env.transition_probs(s, a).items()),
        )
    return V, policy, it


def policy_iteration(env: GridWorld, max_iters: int = 100):
    """Alternate policy evaluation (exact solve) and greedy improvement."""
    states = [s for s in env.states() if s not in (env.goal, env.pit)]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    pi = {s: 0 for s in states}
    for it in range(1, max_iters + 1):
        # policy evaluation: V = R + gamma P V
        P = np.zeros((n, n))
        R = np.zeros(n)
        for s in states:
            R[idx[s]] = env.expected_reward(s, pi[s])
            for nxt, p in env.transition_probs(s, pi[s]).items():
                if nxt in idx:
                    P[idx[s], idx[nxt]] += p
        V_vec = np.linalg.solve(np.eye(n) - env.gamma * P, R)
        V = {s: float(V_vec[idx[s]]) for s in states}
        # improvement
        new_pi = {}
        for s in states:
            new_pi[s] = max(
                range(4),
                key=lambda a: env.expected_reward(s, a)
                + env.gamma * sum(p * V.get(nxt, 0.0) for nxt, p in env.transition_probs(s, a).items()),
            )
        if new_pi == pi:
            return V, pi, it
        pi = new_pi
    return V, pi, max_iters


def td_evaluate(env: GridWorld, policy: dict, episodes: int = 500, alpha: float = 0.1,
                seed: int = 0):
    """TD(0) policy evaluation with a running average."""
    rng = random.Random(seed)
    V = {s: 0.0 for s in env.states()}
    for _ in range(episodes):
        s = env.start
        for _ in range(100):
            a = policy[s]
            s2, r, done = env.step(s, a, rng)
            V[s] += alpha * (r + env.gamma * V[s2] - V[s])
            s = s2
            if done:
                break
    return V


def epsilon_greedy(Q_row: Sequence[float], eps: float, rng) -> int:
    """With probability eps explore uniformly, otherwise act greedily."""
    if rng.random() < eps:
        return rng.randrange(len(Q_row))
    return int(np.argmax(Q_row))


def q_learning(env: GridWorld, episodes: int = 2000, alpha: float = 0.2, gamma: float | None = None,
               eps: float = 0.2, eps_decay: float = 0.999, seed: int = 0):
    """Tabular Q-learning. Returns (Q, policy, returns)."""
    rng = random.Random(seed)
    gamma = env.gamma if gamma is None else gamma
    Q = {s: [0.0] * 4 for s in env.states()}
    returns = []
    for _ in range(episodes):
        s = env.start
        total = 0.0
        for _ in range(100):
            a = epsilon_greedy(Q[s], eps, rng)
            s2, r, done = env.step(s, a, rng)
            target = r + (0.0 if done else gamma * max(Q[s2]))
            Q[s][a] += alpha * (target - Q[s][a])
            total += r
            s = s2
            if done:
                break
        returns.append(total)
        eps *= eps_decay
    policy = {s: int(np.argmax(Q[s])) for s in env.states()}
    return Q, policy, returns


def discounted_return(rewards: Sequence[float], gamma: float = 0.9) -> float:
    return float(sum(gamma**k * r for k, r in enumerate(rewards)))


def regret(returns: Sequence[float], optimal_return: float) -> np.ndarray:
    """Cumulative regret: T * optimal - cumulative reward."""
    r = np.asarray(returns, float)
    return optimal_return * np.arange(1, len(r) + 1) - np.cumsum(r)
