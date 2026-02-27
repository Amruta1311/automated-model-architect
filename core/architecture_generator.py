import torch.nn as nn
import random

# Dynamically create activations from YAML
def get_activation(name):
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU
    }[name]

class ArchitectureGenerator:

    def __init__(self, search_space):
        self.space = search_space

    def sample_architecture(self, input_dim, output_dim):
        # Sample number of layers
        num_layers = random.choice(self.space["layers"]["num_layers"])

        layers = []
        in_dim = input_dim

        # Sample hidden units, activations, and dropout **per layer**
        hidden_units_list = [
            random.choice(self.space["layers"]["hidden_units"]) for _ in range(num_layers)
        ]
        activations_list = [
            random.choice(self.space["layers"]["activation"]) for _ in range(num_layers)
        ]
        dropout_list = [
            random.choice(self.space["layers"]["dropout"]) for _ in range(num_layers)
        ]

        for i in range(num_layers):
            out_dim = hidden_units_list[i]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_activation(activations_list[i])())
            if dropout_list[i] > 0:
                layers.append(nn.Dropout(dropout_list[i]))
            in_dim = out_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        model = nn.Sequential(*layers)

        # Return architecture config for logging
        arch_config = {
            "num_layers": num_layers,
            "hidden_units": hidden_units_list,
            "activation": activations_list,
            "dropout": dropout_list
        }

        return model, arch_config