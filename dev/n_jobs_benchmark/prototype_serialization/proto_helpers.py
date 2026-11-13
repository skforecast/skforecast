"""Tasks of the prototype. In a module, so that the workers keep its cache."""
import pickle

_CACHE = {}


def _predict(forecaster, last_window, exog, steps):
    predictions = forecaster.predict(steps=steps, last_window=last_window, exog=exog)
    return predictions.to_numpy()


def predict_obj(forecaster, last_window, exog, steps):
    return _predict(forecaster, last_window, exog, steps)


def predict_bytes(blob, last_window, exog, steps):
    return _predict(pickle.loads(blob), last_window, exog, steps)


def predict_cached(token, blob, last_window, exog, steps):
    forecaster = _CACHE.get(token)
    if forecaster is None:
        _CACHE.clear()
        forecaster = _CACHE[token] = pickle.loads(memoryview(blob))
    return _predict(forecaster, last_window, exog, steps)
