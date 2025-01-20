# models/model.py

import torch
import torch.nn as nn
import importlib

class MyModel(nn.Module):
    """
    Wrapper model that dynamically imports and initializes a model
    based on the name specified in the config (config.model.name).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # e.g., "resnet50" => imports "models/resnet50.py"
        model_name = self.config['model']['name']  
        try:
            # Dynamically import the module (e.g. models.resnet50)
            mod = importlib.import_module(f"models.{model_name}")
        except ImportError as e:
            raise ImportError(
                f"Could not import module `models.{model_name}`. "
                f"Ensure that a file named `{model_name}.py` exists in the `models` folder."
            ) from e

        # Option 1: If each model file has a class named `Model`
        if hasattr(mod, "Model"):
            model_class = getattr(mod, "Model")
            self.net = model_class(self.config)
        # Option 2: Or if each model file has a function named `build_model`
        elif hasattr(mod, "build_model"):
            build_model_func = getattr(mod, "build_model")
            self.net = build_model_func(self.config.model)
        else:
            raise AttributeError(
                f"Module `models.{model_name}` must define a class named `Model` or "
                f"a function named `build_model`."
            )

    def forward(self, x):
        return self.net(x)
