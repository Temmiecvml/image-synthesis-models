import importlib


def instantiate_object(config, reload=False):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    module, cls = config["target"].rsplit(".", 1)

    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)

    params = config.get("params", dict())
    object = getattr(module_imp, cls)(**params)

    return object
