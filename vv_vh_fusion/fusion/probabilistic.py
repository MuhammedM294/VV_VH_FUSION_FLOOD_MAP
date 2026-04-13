import numpy as np


_EPS = 1e-6


def _valid_from_flood(flood: np.ndarray) -> np.ndarray:
    """Valid pixels are exactly those with a real decision (0/1)."""
    return (flood == 0) | (flood == 1)


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(p, _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fuse_bayesian_product(
    flood_vv: np.ndarray,
    flood_vh: np.ndarray,
    prob_vv: np.ndarray,
    prob_vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    B1 - Bayesian product under (approx.) conditional independence:
        logit(P_fused) = logit(Pvv) + logit(Pvh)

    Returns
    -------
    p_fused : float32 array, [0,1]
    valid   : bool mask (VV or VH valid)
    """
    vv_valid = _valid_from_flood(flood_vv)
    vh_valid = _valid_from_flood(flood_vh)
    valid = vv_valid | vh_valid

    p_fused = np.full_like(prob_vv, np.nan, dtype=np.float32)

    # Cases:
    # - both valid: combine
    both = vv_valid & vh_valid
    if np.any(both):
        p_fused[both] = _sigmoid(_logit(prob_vv[both]) + _logit(prob_vh[both]))

    # - only one valid: fall back to that sensor
    vv_only = vv_valid & ~vh_valid
    if np.any(vv_only):
        p_fused[vv_only] = prob_vv[vv_only].astype(np.float32)

    vh_only = vh_valid & ~vv_valid
    if np.any(vh_only):
        p_fused[vh_only] = prob_vh[vh_only].astype(np.float32)

    return p_fused, valid


def fuse_weighted_logit(
    flood_vv: np.ndarray,
    flood_vh: np.ndarray,
    prob_vv: np.ndarray,
    prob_vh: np.ndarray,
    w_vv: float = 0.5,
    w_vh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    B2 - Weighted logit fusion:
        logit(P_fused) = w_vv*logit(Pvv) + w_vh*logit(Pvh)

    Notes
    -----
    - Use weights to reduce correlation overconfidence.
    - If only one sensor is valid, fallback to that sensor.
    """
    vv_valid = _valid_from_flood(flood_vv)
    vh_valid = _valid_from_flood(flood_vh)
    valid = vv_valid | vh_valid

    p_fused = np.full_like(prob_vv, np.nan, dtype=np.float32)

    both = vv_valid & vh_valid
    if np.any(both):
        x = (w_vv * _logit(prob_vv[both])) + (w_vh * _logit(prob_vh[both]))
        p_fused[both] = _sigmoid(x)

    vv_only = vv_valid & ~vh_valid
    if np.any(vv_only):
        p_fused[vv_only] = prob_vv[vv_only].astype(np.float32)

    vh_only = vh_valid & ~vv_valid
    if np.any(vh_only):
        p_fused[vh_only] = prob_vh[vh_only].astype(np.float32)

    return p_fused, valid


def fuse_entropy_weighted_average(
    flood_vv: np.ndarray,
    flood_vh: np.ndarray,
    prob_vv: np.ndarray,
    prob_vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    B3 - Entropy-weighted probability averaging:
        w = 1/(H(p)+eps),  P = (wvv*Pvv + wvh*Pvh)/(wvv+wvh)

    Lower entropy => higher weight.
    """
    vv_valid = _valid_from_flood(flood_vv)
    vh_valid = _valid_from_flood(flood_vh)
    valid = vv_valid | vh_valid

    p_fused = np.full_like(prob_vv, np.nan, dtype=np.float32)

    # entropy helper (binary entropy)
    def H(p):
        p = _clip_prob(p)
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))

    both = vv_valid & vh_valid
    if np.any(both):
        pvv = prob_vv[both]
        pvh = prob_vh[both]
        wvv = 1.0 / (H(pvv) + _EPS)
        wvh = 1.0 / (H(pvh) + _EPS)
        p_fused[both] = ((wvv * pvv) + (wvh * pvh)) / (wvv + wvh)

    vv_only = vv_valid & ~vh_valid
    if np.any(vv_only):
        p_fused[vv_only] = prob_vv[vv_only].astype(np.float32)

    vh_only = vh_valid & ~vv_valid
    if np.any(vh_only):
        p_fused[vh_only] = prob_vh[vh_only].astype(np.float32)

    return p_fused, valid


def probability_to_binary(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probability to a binary flood map.
    Returns uint8 with {0,1}; NaNs remain 0 unless caller masks them out.
    """
    out = np.zeros_like(p, dtype=np.uint8)
    m = np.isfinite(p)
    out[m] = (p[m] >= threshold).astype(np.uint8)
    return out