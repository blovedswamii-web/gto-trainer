"""Numba kernels for the batched solver's hot loops (optional dependency).

Each kernel is a straight loop over (board, hand), parallelised over boards with ``prange``
so all cores are used, and fuses what NumPy would do in several passes over memory:

* ``river_showdown``  -- prefix sums of opponent reach in rank order (and per card group),
  then per hero hand: weaker - stronger - card-removal correction. float64 accumulators.
* ``compat_sum``      -- opponent reach compatible with each hero hand via per-card sums.
* ``regret_match``    -- strategy proportional to positive regret (uniform if none).
* ``normalize``       -- average strategy from the strategy sum (uniform if empty).
* ``dcfr_update``     -- node value, instantaneous regret, discounted regret and
  average-strategy update in one pass, in place.

If Numba isn't installed, ``HAVE_NUMBA`` is False and the solver uses its NumPy code
paths, which compute exactly the same quantities (the tests check both against each other).
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly
    from numba import njit, prange

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


if HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def river_showdown(ro, order, lo, hi, e_combo, c_lo, c_hi, c_start, c_end, amount):  # type: ignore[no-untyped-def]
        nb, no = ro.shape
        nh = lo.shape[1]
        ne = e_combo.shape[1]
        out = np.empty((nb, nh), np.float32)
        for b in prange(nb):
            cum = np.empty(no + 1)
            cum[0] = 0.0
            for k in range(no):
                cum[k + 1] = cum[k] + ro[b, order[b, k]]
            ce = np.empty(ne + 1)
            ce[0] = 0.0
            for k in range(ne):
                ce[k + 1] = ce[k] + ro[b, e_combo[b, k]]
            total = cum[no]
            for i in range(nh):
                v = cum[lo[b, i]] - (total - cum[hi[b, i]])
                for c in range(2):
                    weaker = ce[c_lo[c, b, i]] - ce[c_start[c, b, i]]
                    stronger = ce[c_end[c, b, i]] - ce[c_hi[c, b, i]]
                    v -= weaker - stronger
                out[b, i] = v * amount
        return out

    @njit(parallel=True, cache=True)
    def compat_sum(ro, hero_cards, opp_cards, same):  # type: ignore[no-untyped-def]
        nb, no = ro.shape
        nh = hero_cards.shape[0]
        out = np.empty((nb, nh), np.float32)
        for b in prange(nb):
            cs = np.zeros(52)
            tot = 0.0
            for j in range(no):
                r = ro[b, j]
                tot += r
                cs[opp_cards[j, 0]] += r
                cs[opp_cards[j, 1]] += r
            for i in range(nh):
                s = cs[hero_cards[i, 0]] + cs[hero_cards[i, 1]]
                if same[i] < no:
                    s -= ro[b, same[i]]
                out[b, i] = tot - s
        return out

    @njit(parallel=True, cache=True)
    def regret_match(regret):  # type: ignore[no-untyped-def]
        nb, n, na = regret.shape
        out = np.empty_like(regret)
        for b in prange(nb):
            for i in range(n):
                tot = 0.0
                for a in range(na):
                    r = regret[b, i, a]
                    if r > 0:
                        tot += r
                for a in range(na):
                    if tot > 0:
                        r = regret[b, i, a]
                        out[b, i, a] = r / tot if r > 0 else 0.0
                    else:
                        out[b, i, a] = 1.0 / na
        return out

    @njit(parallel=True, cache=True)
    def normalize(strat_sum):  # type: ignore[no-untyped-def]
        nb, n, na = strat_sum.shape
        out = np.empty_like(strat_sum)
        for b in prange(nb):
            for i in range(n):
                tot = 0.0
                for a in range(na):
                    tot += strat_sum[b, i, a]
                for a in range(na):
                    out[b, i, a] = strat_sum[b, i, a] / tot if tot > 0 else 1.0 / na
        return out

    @njit(parallel=True, cache=True)
    def dcfr_update(regret, strat_sum, sigma, q, rp, pos_d, neg_d, strat_d, cfr_plus, t):  # type: ignore[no-untyped-def]
        nb, n, na = regret.shape
        out = np.empty((nb, n), np.float32)
        for b in prange(nb):
            for i in range(n):
                v = 0.0
                for a in range(na):
                    v += sigma[b, i, a] * q[b, i, a]
                w = rp[b, i]
                for a in range(na):
                    inst = q[b, i, a] - v
                    r = regret[b, i, a]
                    if cfr_plus:
                        nr = r + inst
                        regret[b, i, a] = nr if nr > 0 else 0.0
                        strat_sum[b, i, a] += w * sigma[b, i, a] * t
                    else:
                        regret[b, i, a] = (r * pos_d if r > 0 else r * neg_d) + inst
                        strat_sum[b, i, a] = strat_sum[b, i, a] * strat_d + w * sigma[b, i, a]
                out[b, i] = v
        return out
