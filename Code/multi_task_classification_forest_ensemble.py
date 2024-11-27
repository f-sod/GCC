import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import itertools
from sklearn.metrics import matthews_corrcoef


class MultiTaskForestEnsemble:

    def __init__(self, n_forests: int, grid_search: bool = True, n_tree_range: list[int] = [100], criterion: str = 'gini', max_depth_range: list[int] = [None],
                 min_sample_split_range: list[int] = [2], min_samples_leaf_range: list[int] = [1], min_weight_fraction_leaf: float = 0.0,
                 max_features: str = 'sqrt', max_leaf_node_range: list[int] = [None], min_impurity_decrease_range: list[float] = [0.0],
                 bootstrap: bool = True, oob_score: bool = False, n_jobs: int = -1, random_state: int = 42, verbose: int = 0,
                 warm_start: bool = False, class_weight: str = None, ccp_alpha: float = 0.0, cv: int = 5, score_for_cv: str = 'balanced_accuracy'
                 ):
        '''
            n_forest many RF classifier from sklearn will be created

            the parameters in the specified range will be tuned using the given parameter ranges
            if no parameter range is indicated, the classifiers will be instantiated with the same default values as in
            the sklearn package
        '''
        self.n_forets = n_forests
        self.grid_search = grid_search
        self.n_tree_range = n_tree_range
        self.criterion = criterion
        self.max_depth_range = max_depth_range
        self.min_sample_split_range = min_sample_split_range
        self.min_sample_leaf_range = min_samples_leaf_range
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_node_range = max_leaf_node_range
        self.min_impurity_decrease_range = min_impurity_decrease_range
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.cv = cv
        self.score_for_cv = score_for_cv
        if self.grid_search:
            self.param_grid = {'max_depth': self.max_depth_range,
                               'min_samples_split': self.min_sample_split_range,
                               'min_samples_leaf': self.min_sample_leaf_range,
                               'max_leaf_nodes': self.max_leaf_node_range,
                               'min_impurity_decrease': self.min_impurity_decrease_range}

    def cross_validate_and_fit_forests(self, X_train: pd.DataFrame, y_train: pd.DataFrame, train_indices: dict, train_feature_list: list[list] = None, cv_feature_list: list[list] = None, cv_folds: list[tuple] = None):
        self.forests_ = []
        for forest_n in range(self.n_forets):
            X_train_cleaned = X_train.loc[train_indices[forest_n], :]
            y_train_cleaned = y_train.iloc[forest_n, :]
            y_train_cleaned = y_train_cleaned[train_indices[forest_n]]
            if self.grid_search:
                if cv_folds is None:
                    gs = GridSearchCV(RandomForestClassifier(
                        random_state=self.random_state, criterion=self.criterion, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                        max_features=self.max_features, bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs,
                        verbose=self.verbose, warm_start=self.warm_start, class_weight=self.class_weight, ccp_alpha=self.ccp_alpha),
                        param_grid=self.param_grid, refit=True, cv=self.cv, scoring=self.score_for_cv)
                    gs.fit(X_train_cleaned, y_train_cleaned)
                    rf = gs.best_estimator_
                else:
                    keys, values = zip(*self.param_grid.items())
                    permutations_dicts = [dict(zip(keys, v))
                                          for v in itertools.product(*values)]
                    rf = RandomForestClassifier(
                        random_state=self.random_state, criterion=self.criterion, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                        max_features=self.max_features, bootstrap=self.bootstrap, oob_score=self.oob_score, n_jobs=self.n_jobs,
                        verbose=self.verbose, warm_start=self.warm_start, class_weight=self.class_weight, ccp_alpha=self.ccp_alpha)

                    best_score = -1
                    best_estimator = None
                    for permutation in permutations_dicts:
                        current_rf = rf.set_params(**permutation)
                        folds = cv_folds[forest_n]
                        fold_score_arr = []

                        for fold in folds:
                            print(fold)
                            X_fold = X_train_cleaned.iloc[fold[0]]
                            y_fold = y_train_cleaned.iloc[fold[0]]
                            current_rf.fit(X_fold, y_fold)
                            y_pred_fold = current_rf.predict(
                                X_train_cleaned.iloc[fold[1]])
                            fold_score_arr.append(matthews_corrcoef(
                                y_train_cleaned.iloc[fold[1]], y_pred_fold))
                        new_score = np.mean(fold_score_arr)
                        if new_score > best_score:
                            best_score = new_score
                            best_estimator = current_rf
                    rf = best_estimator
                if train_feature_list is None:
                    rf.fit(X_train_cleaned, y_train_cleaned)
                else:
                    X_train_cleaned = X_train_cleaned.loc[:,
                                                          train_feature_list[forest_n]]
                    rf.fit(X_train_cleaned, y_train_cleaned)
                self.forests_.append(rf)
            else:
                pass

    def predict(self, X_test, test_indices: dict, train_feature_list: list[list] = None):
        predictions = []
        for n, forest_n in enumerate(self.forests_):
            if train_feature_list is None:
                y_pred_n = forest_n.predict(X_test.loc[test_indices[n]])
            else:
                y_pred_n = forest_n.predict(
                    X_test.loc[test_indices[n], train_feature_list[n]])
            predictions.append(y_pred_n)
        return predictions
