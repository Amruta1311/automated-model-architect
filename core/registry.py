class ModelRegistry:

    def __init__(self):
        self.results = []

    def log(self, config, accuracy):
        self.results.append({
            "config": config,
            "accuracy": accuracy
        })

    def get_best(self):
        return max(self.results, key=lambda x: x["accuracy"])