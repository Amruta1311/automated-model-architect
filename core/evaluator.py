import torch
from sklearn.metrics import accuracy_score

class Evaluator:

    def evaluate(self, model, X_test, y_test):

        model.eval()
        with torch.no_grad():
            preds = model(X_test)
            predicted = torch.argmax(preds, dim=1)

        acc = accuracy_score(y_test.numpy(), predicted.numpy())
        return acc