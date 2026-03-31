from .rate import RateEncoder, RateDecoder
from .temporal import TemporalEncoder, PhaseEncoder
from .population import PopulationEncoder, DeltaEncoder

__all__ = [
    "RateEncoder", "RateDecoder",
    "TemporalEncoder", "PhaseEncoder",
    "PopulationEncoder", "DeltaEncoder",
]


def encode(x, method="rate", **kwargs):
    """
    Convenience function to encode data with the specified method.

    Parameters
    ----------
    x : np.ndarray
    method : str
        'rate', 'temporal', 'population', or 'delta'
    **kwargs : passed to the encoder

    Returns
    -------
    spike_train : np.ndarray
    """
    encoders = {
        "rate": RateEncoder,
        "temporal": TemporalEncoder,
        "population": PopulationEncoder,
        "delta": DeltaEncoder,
    }
    if method not in encoders:
        raise ValueError(f"Unknown encoding method: {method}. Choose from {list(encoders.keys())}")
    encoder = encoders[method](**kwargs)
    return encoder.encode(x)
