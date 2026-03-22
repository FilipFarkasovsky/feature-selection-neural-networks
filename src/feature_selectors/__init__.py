from .cancelout import CancelOutFeatureSelector
from .decision_tree import DecisionTreeFeatureSelector
from .kruskall_wallis_filter import KruskalWallisFeatureSelector
from .lasso import LassoFeatureSelector
from .lassonet import LassoNetFeatureSelector
from .deeppink import Deeppink
from .cae import CAEFeatureSelector
from .fsnet import FSNetFeatureSelector
from .linear_svm import LinearSVMFeatureSelector
from .logistic_regression_forward_selector import LRForwardFeatureSelector
from .logistic_regression_genetic_algorithm import LRGAFeatureSelector
from .mrmr import MRMRFeatureSelector
from .mrmr_ga import MRMRGAFeatureSelector
from .mutual_info_filter import MutualInformationFeatureSelector
from .random_forest import RandomForestFeatureSelector
from .relieff import ReliefFFeatureSelector
from .relieff_ga import ReliefFGAFeatureSelector
from .ridge import RidgeClassifierFeatureSelector
from .svm_forward_selector import SVMForwardFeatureSelector
from .svm_genetic_algorithm import SVMGAFeatureSelector
from .svm_rfe import SVMRFE

feature_selectors = {
    "SVMGeneticAlgorithm": SVMGAFeatureSelector,
    "LogisticRegressionGeneticAlgorithm": LRGAFeatureSelector,
    "ReliefFGeneticAlgorithm": ReliefFGAFeatureSelector,
    "MRMRGeneticAlgorithm": MRMRGAFeatureSelector,

    "Cancelout": CancelOutFeatureSelector,
    "DecisionTree": DecisionTreeFeatureSelector,
    "KruskallWallisFilter": KruskalWallisFeatureSelector,
    "Lasso": LassoFeatureSelector,
    "LassoNet": LassoNetFeatureSelector,
    "Deeppink": Deeppink,
    "CAE": CAEFeatureSelector,
    "FSNet" : FSNetFeatureSelector,
    "LinearSVM": LinearSVMFeatureSelector,
    "LRFowardSelection": LRForwardFeatureSelector,
    "MRMR": MRMRFeatureSelector,
    "MutualInformationFilter": MutualInformationFeatureSelector,
    "RandomForest": RandomForestFeatureSelector,
    "ReliefFFeatureSelector": ReliefFFeatureSelector,
    "RidgeClassifier": RidgeClassifierFeatureSelector,
    "SVMFowardSelection": SVMForwardFeatureSelector,
    "SVMRFE": SVMRFE,
}
