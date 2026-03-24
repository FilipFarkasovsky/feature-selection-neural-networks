from .cancelout import CancelOutFeatureSelector
from .decision_tree import DecisionTreeFeatureSelector
from .kruskall_wallis_filter import KruskalWallisFeatureSelector
from .lasso import LassoFeatureSelector
from .lassonet import LassoNetFeatureSelector
from .deeppink import Deeppink
from .cae import CAEFeatureSelector
from .fsnet import FSNetFeatureSelector
from .linear_svm import LinearSVMFeatureSelector
from .mrmr import MRMRFeatureSelector
from .mutual_info_filter import MutualInformationFeatureSelector
from .random_forest import RandomForestFeatureSelector
from .relieff import ReliefFFeatureSelector
from .svm_forward_selector import SVMForwardFeatureSelector
from .svm_rfe import SVMRFE

feature_selectors = {
    "Cancelout": CancelOutFeatureSelector,
    "DecisionTree": DecisionTreeFeatureSelector,
    "KruskallWallisFilter": KruskalWallisFeatureSelector,
    "Lasso": LassoFeatureSelector,
    "LassoNet": LassoNetFeatureSelector,
    "Deeppink": Deeppink,
    "CAE": CAEFeatureSelector,
    "FSNet" : FSNetFeatureSelector,
    "LinearSVM": LinearSVMFeatureSelector,
    "MRMR": MRMRFeatureSelector,
    "MutualInformationFilter": MutualInformationFeatureSelector,
    "RandomForest": RandomForestFeatureSelector,
    "ReliefFFeatureSelector": ReliefFFeatureSelector,
    "SVMFowardSelection": SVMForwardFeatureSelector,
    "SVMRFE": SVMRFE,
}
