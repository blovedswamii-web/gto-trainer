"""Closed-form betting maths used by the odds tools and drills.

Conventions: ``pot`` is the pot *before* the bet in question; ``bet`` is the size of the bet
being faced (or made). All results are fractions in [0, 1].
"""

from __future__ import annotations


def _check(pot: float, bet: float) -> None:
    if pot <= 0 or bet <= 0:
        raise ValueError("pot and bet must be positive")


def pot_odds(pot: float, bet: float) -> float:
    """Equity a caller needs to break even: ``call / final pot = bet / (pot + 2*bet)``.

    Ignores future streets (i.e. assumes the hand is checked down / all-in after calling).
    """
    _check(pot, bet)
    return bet / (pot + 2 * bet)


def minimum_defence_frequency(pot: float, bet: float) -> float:
    """MDF = ``pot / (pot + bet)``.

    If the defender continues less often than this, a bet with *any two cards* profits
    immediately, because the bettor risks ``bet`` to win ``pot``.
    """
    _check(pot, bet)
    return pot / (pot + bet)


def bluff_breakeven(pot: float, bet: float) -> float:
    """How often a pure bluff must work: ``bet / (pot + bet)`` (= 1 - MDF), often called alpha."""
    _check(pot, bet)
    return bet / (pot + bet)


def optimal_bluff_fraction(pot: float, bet: float) -> float:
    """Fraction of a polarised river betting range that should be bluffs.

    Makes the caller indifferent: caller risks ``bet`` to win ``pot + bet``, so bluffs must be
    ``bet / (pot + 2*bet)`` of the betting range -- numerically equal to the caller's pot odds.
    """
    _check(pot, bet)
    return bet / (pot + 2 * bet)


def value_to_bluff_ratio(pot: float, bet: float) -> float:
    """Value combos per bluff combo for a polarised river bet: ``(pot + bet) / bet``."""
    _check(pot, bet)
    return (pot + bet) / bet
