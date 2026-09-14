"""Behavioural metrics computed from evaluation logs.

Modules:
    routine         clock information, entropy rate, predictability, periodicity, shock recovery
    specialisation  division of labour, specialisation index, clustering, identity swap test
    social          co-presence networks, association, synchrony, contention, gossip diffusion
    stats           interquartile mean, bootstrap intervals, paired tests, Holm correction
    dashboard       degeneracy flags and one-row episode summaries
"""
from hamlet.metrics import dashboard, routine, social, specialisation, stats

__all__ = ["dashboard", "routine", "social", "specialisation", "stats"]
