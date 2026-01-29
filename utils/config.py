import yaml

def load_config(paths):
    cfg = {}
    for p in paths:
        with open(p, "r") as f:
            part = yaml.safe_load(f)
            cfg.update(part)
    return cfg