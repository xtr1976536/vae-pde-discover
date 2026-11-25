import json

from torch import nn


def network(widths, activation):
    modules = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        linear = nn.Linear(left, right)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        modules.append(linear)
        if i < len(widths) - 2:
            modules.append(activation())
    return nn.Sequential(*modules)


def atomic_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
