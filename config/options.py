import os
import json
import argparse
import copy

class Options:
    def __init__(self):
        self.default_config = {
            "batchsize": 16,
            "epochs": 100,
            "train_type": "train",
            "dataset_version": "v1",
            "save_path": "./results",
            "folder_path": "./data",
            "w_con": 1.0,
            "w_adv": 1.0,
            "w_enc": 1.0,
            "model_architecture": {
                "encoder_cbam": False,
                "decoder_cbam": False
            }
        }

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default=None)
        args, unknown_args = parser.parse_known_args()

        config_path = args.config or os.path.join(os.path.dirname(__file__), 'config.json')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        merged = copy.deepcopy(self.default_config)
        merged.update(config_data)

        # Parse unknown args: e.g., --con_loss l2 --> {"con_loss": "l2"}
        cli_overrides = self._parse_unknown_args(unknown_args)
        merged = self._apply_overrides(merged, cli_overrides)

        return self._dict_to_obj(merged)

    def _parse_unknown_args(self, args_list):
        overrides = {}
        key = None
        for arg in args_list:
            if arg.startswith('--'):
                key = arg[2:]
            else:
                if key:
                    overrides[key] = self._auto_type(arg)
                    key = None
        return overrides

    def _apply_overrides(self, config, overrides):
        for key, value in overrides.items():
            if '.' in key:  # Soporta cosas como model_architecture.encoder_cbam
                keys = key.split('.')
                d = config
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
            else:
                config[key] = value
        return config

    def _auto_type(self, val):
        if val.lower() == 'true':
            return True
        elif val.lower() == 'false':
            return False
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def _dict_to_obj(self, d):
        class ConfigObject:
            def __init__(self, entries):
                for k, v in entries.items():
                    if isinstance(v, dict):
                        setattr(self, k, Options._dict_to_obj_static(v))
                    else:
                        setattr(self, k, v)
        return ConfigObject(d)

    @staticmethod
    def _dict_to_obj_static(d):
        class ConfigObject:
            def __init__(self, entries):
                for k, v in entries.items():
                    if isinstance(v, dict):
                        setattr(self, k, Options._dict_to_obj_static(v))
                    else:
                        setattr(self, k, v)
        return ConfigObject(d)
