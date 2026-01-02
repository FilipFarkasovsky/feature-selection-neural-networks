from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import minmax_scale

from .scoring import default_scoring
from .models import default_models


class SelectionScorer:
    @staticmethod
    def _eval(X, y, model, scoring):
        X = minmax_scale(X)
        cv = StratifiedKFold()

        results = cross_validate(model, X, y, cv=cv, scoring=scoring)

        avg_results = {
            k.replace("test_", ""): v.mean()
            for (k, v) in results.items()
            if not k.endswith("time")
        }

        return avg_results
    
    @staticmethod
    def eval(X, y, models=default_models, scoring=default_scoring):
        return {name: SelectionScorer._eval(X, y, model, scoring) for name, model in models.items()}
