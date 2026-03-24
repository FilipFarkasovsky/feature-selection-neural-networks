from feature_selectors.base_models.base_selector import BaseSelector


class Task():
    def __init__(
        self,
        name: str,
        feature_selector: BaseSelector,
        dataset_name: str,
        n_informative: int,
        sampling: str = 'none'
    ):
        self.name = name
        self.feature_selector = feature_selector
        self.dataset_name = dataset_name
        self.n_informative = n_informative
        self.sampling = sampling
