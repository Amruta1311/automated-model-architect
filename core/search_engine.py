import yaml
import torch
import random  # needed for sampling
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from .architecture_generator import ArchitectureGenerator
from .trainer import Trainer
from .evaluator import Evaluator
from .registry import ModelRegistry

class SearchEngine:

    def __init__(self, search_space_path):
        with open(search_space_path, "r") as f:
            self.search_space = yaml.safe_load(f)

        self.generator = ArchitectureGenerator(self.search_space)
        self.trainer = Trainer()
        self.evaluator = Evaluator()
        self.registry = ModelRegistry()

    def run(self, n_trials=5):

        data = load_iris()
        X = torch.tensor(data.data, dtype=torch.float32)
        y = torch.tensor(data.target, dtype=torch.long)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        input_dim = X.shape[1]
        output_dim = len(set(y.numpy()))

        trial_results = []

        for trial in range(n_trials):

            model, arch_config = self.generator.sample_architecture(
                input_dim, output_dim
            )

            train_config = {
                "lr": 0.001,
                "batch_size": 32,
                "epochs": self.search_space["training"]["epochs"]
            }
            # Sample training hyperparameters from search space
            train_config = {
                "lr": random.choice(self.search_space["training"]["lr"]),
                "batch_size": random.choice(self.search_space["training"]["batch_size"]),
                "epochs": self.search_space["training"]["epochs"]
            }

            model = self.trainer.train(model, X_train, y_train, train_config)

            acc = self.evaluator.evaluate(model, X_test, y_test)

            full_config = {**arch_config, **train_config}

            result = {
                "trial": trial + 1,
                "accuracy": acc,
                **full_config
            }

            trial_results.append(result)
            self.registry.log(full_config, acc)

        best = max(trial_results, key=lambda x: x["accuracy"])

        return {
            "best": best,
            "all_trials": trial_results
        }