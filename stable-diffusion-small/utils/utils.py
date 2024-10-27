import importlib


def instantiate_object(config, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    module, cls = config["target"].rsplit(".", 1)

    module_imp = importlib.import_module(module)
    params = config.get("params", dict())
    params.update(kwargs)
    object = getattr(module_imp, cls)(**params)

    return object


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
