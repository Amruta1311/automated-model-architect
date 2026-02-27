from core.search_engine import SearchEngine

engine = SearchEngine("configs/search_space.yaml")

def run_search(trials: int = 5):
    return engine.run(n_trials=trials)