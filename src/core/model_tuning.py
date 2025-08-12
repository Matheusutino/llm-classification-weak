# model_selector.py
import numpy as np
from skopt import BayesSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Union
from sklearn.metrics._scorer import _Scorer


class ModelTuning:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = self._select_model()
        self.param_grid = self._select_hyperparameters()

    def _select_model(self) -> BaseEstimator:
        if self.model_name == "knn":
            model =  KNeighborsClassifier(n_jobs=1)
        elif self.model_name == "random_forest":
            model = RandomForestClassifier(n_jobs=-1)
        else:
            raise ValueError("Unsupported model type.")
        
        return model

    def _select_hyperparameters(self) -> Dict[str, Union[str, int]]:
        if self.model_name == "knn":
            param_grid = {
                'n_neighbors': list(range(3, 51)),             
                'metric': ['euclidean', 'manhattan', 'cosine'],  
                'weights': ['uniform', 'distance']                 
            }
        elif self.model_name == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            }
        else:
            raise ValueError("Unsupported model type.")
        
        return param_grid

    def _select_scoring(self) -> Dict[str, _Scorer]:
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, average="macro"),
            "recall": make_scorer(recall_score, average="macro"),
            "f1_score": make_scorer(f1_score, average="macro")
        }

        return scoring


    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, metric_optmize: str = "f1_score", n_iter: int = 10, cv: int = 5, random_state: int = 42):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        bayes_search = BayesSearchCV(self.model, 
                                     self.param_grid, 
                                     n_iter=n_iter, 
                                     cv=cv, 
                                     scoring=self._select_scoring(), 
                                     random_state=random_state, 
                                     return_train_score=True, 
                                     refit=metric_optmize,
                                     n_jobs=5)
        bayes_search.fit(X, y)

        return bayes_search
