from __future__ import annotations


def cubic_like_baseline(cwnd: float, loss_event: bool = False) -> float:
    """Small userspace approximation for controller simulations."""
    cwnd = max(float(cwnd), 1.0)
    if loss_event:
        return max(1.0, cwnd * 0.7)
    return cwnd + 1.0 / cwnd

