"""Independent Chip Model (Malmuth-Harville) tournament equity.

Model: the probability that a player finishes 1st is proportional to their stack; given the
players already placed, the next place is again drawn proportionally to the remaining stacks.
So for an ordered finish (a, b, c, ...)::

    P = s_a / T  *  s_b / (T - s_a)  *  s_c / (T - s_a - s_b)  * ...

Summing over orders is factorial, but the probability of reaching a given *set* of placed
players doesn't depend on their order beyond its total, so a DP over subsets works:
``prob[S]`` = P(the top |S| places are exactly the players in S). Each step places one more
player ``i`` with probability ``s_i / (T - sum(S))`` and pays them ``payouts[|S|]``. Cost is
O(sum_{m<k} C(n, m) * n) for k paid places -- instant for a 9-player final table.

Busted players (stack 0) take the bottom places, so a bust in the current hand gets paid
the lowest place among the players who started the hand.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Arr = npt.NDArray[np.float64]


def icm_equity(stacks: Sequence[float] | Arr, payouts: Sequence[float]) -> Arr:
    """Expected payout of each player (same units as ``payouts``)."""
    s = np.asarray(stacks, dtype=np.float64)
    n = len(s)
    pay = list(payouts) + [0.0] * max(0, n - len(payouts))
    eq = np.zeros(n)
    alive = [i for i in range(n) if s[i] > 1e-12]
    busted = sorted((i for i in range(n) if s[i] <= 1e-12), key=lambda i: -s[i])
    # bottom places go to busted players (only one can bust per all-in in this model)
    for k, i in enumerate(busted):
        eq[i] += pay[n - 1 - k] if n - 1 - k < len(pay) else 0.0
    m = len(alive)
    if m == 0:
        return eq
    # Places worth paying among the alive players are 0 .. m-1; only those with payouts matter.
    last_paid = max((p for p in range(m) if pay[p] != 0.0), default=-1)
    if last_paid < 0:
        return eq
    sa = s[alive]
    total = float(sa.sum())
    probs: dict[int, float] = {0: 1.0}  # bitmask over `alive` index -> probability
    for place in range(last_paid + 1):
        nxt: dict[int, float] = {}
        for mask, p in probs.items():
            used = sum(sa[b] for b in range(m) if mask >> b & 1)
            rem = total - used
            if rem <= 1e-12:
                continue
            for b in range(m):
                if mask >> b & 1:
                    continue
                q = p * sa[b] / rem
                eq[alive[b]] += q * pay[place]
                nm = mask | (1 << b)
                nxt[nm] = nxt.get(nm, 0.0) + q
        probs = nxt
    return eq


def bubble_factor(
    stacks: Sequence[float], payouts: Sequence[float], i: int, j: int, amount: float | None = None
) -> float:
    """Risk premium of player i vs j: equity lost by losing ``amount`` to j divided by equity
    gained by winning it. 1.0 = chip EV; > 1 means i needs better than pot odds to call."""
    s = np.asarray(stacks, dtype=np.float64)
    x = float(min(s[i], s[j]) if amount is None else amount)
    base = icm_equity(s, payouts)[i]
    win, lose = s.copy(), s.copy()
    win[i] += x
    win[j] -= x
    lose[i] -= x
    lose[j] += x
    gain = icm_equity(win, payouts)[i] - base
    loss = base - icm_equity(lose, payouts)[i]
    return float(loss / gain) if gain > 1e-12 else float("inf")
