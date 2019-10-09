from ast import literal_eval


def override_config(config, options):
    for k, v in zip(options[0::2], options[1::2]):
        assert k.startswith('--')

        keys = k[2:].split('.')
        d = config
        for subkey in keys[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = keys[-1]

        try:
            value = literal_eval(v)
        except Exception:
            value = v
        assert isinstance(value, type(d[subkey]))
        d[subkey] = value
    return config
