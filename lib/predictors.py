import time
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error,
                             mean_absolute_percentage_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             plot_roc_curve, plot_precision_recall_curve,
                             plot_confusion_matrix)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split, cross_val_score)
from sklearn.tree import (DecisionTreeClassifier,
                          DecisionTreeRegressor, plot_tree)
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import lib.constants as c
from typing import Any
# -----------------------------------------------------------------------------


class Predictor:
    """Tree-like models training."""

    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, Y_type: str,
                 rnd_state=0, seed=0.33, debug=0):
        """Initialize default models hyperparams and attributes.

        Store training dataset, target class (variable)
        as class attributes for using in training, calc metrics, and so on.
        (it's expensive, but whatever)

        Parameters
        ----------
        X, Y : pd.DataFrame
            Training dataset and target class(variable).

        debug : int, default = 0
            verbosity level

        X, Y : pd.DataFrame
            Training dataset and target class(variable).

        Y_type : {"clf", "regr"}
            solving classification or regression problem,
            i.e. determine if Y is a target class or a variable.

        seed, rnd_state: used for train_test_split
            if not defined, rnd_state generate from [1..500]

        Methods
        -------
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        self.search_type = None
        self.X = X
        self.Y = Y
        self.Y_type = Y_type
        self.debug = debug
        self.tree_clf_criterion = {'criterion': ['gini', 'entropy']}
        self.tree_regr_criterion = {'criterion': [
            "squared_error", "friedman_mse", "absolute_error", "poisson"]}
        self.tree_params = {'max_depth': range(10, 150),
                            'max_features': range(10, 100),
                            'min_samples_split': range(5, 20),
                            'min_samples_leaf': range(5, 20),
                            'max_leaf_nodes': range(10, 400)}
        self.forest_params = {'n_estimators': [10, 15, 20, 30, 50, 70, 100, 120, 150]}
        self.forest_regr_criterion = {'criterion': [
            "squared_error", "absolute_error", "poisson"]}
        self.ada_params = {'n_estimators': [10, 15, 20, 30, 50, 70, 100, 120, 150],
                           "learning_rate": [0.1, 0.01, 0.2, 0.05, 0.3, 0.5, 0.7, 1]}
        self.ada_regr_criterion = {'loss': ['linear', 'square', 'exponential']}
        self.extra_regr_criterion = {
            'criterion': ["squared_error", "absolute_error"]}
        self.gbdt_clf_criterion = {
            'criterion': ['friedman_mse', 'squared_error'],
            'loss': ['log_loss', 'exponential']}
        self.gbdt_regr_criterion = {
            'criterion': ['friedman_mse', 'squared_error'],
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']}
        self.seed = seed
        self.rnd_state = np.random.randint(
            1, 500) if rnd_state == 0 else rnd_state
        self.bayes_search_space = {
            "max_depth": Integer(5, 100),
            "max_features": Categorical(['auto', 'sqrt', 'log2']),
            "min_samples_leaf": Integer(5, 20),
            "min_samples_split": Integer(5, 20),
            "n_estimators": Integer(10, 150)}

    def __get_model_type(self):
        """Define model_type by model from Class attributes."""
        model = self.trained_model

        tree_list = ['DecisionTreeClassifier', 'DecisionTreeRegressor']
        forest_list = ['RandomForestClassifier', 'RandomForestRegressor']
        ada_list = ['AdaBoostClassifier', 'AdaBoostRegressor']
        extra_list = ['ExtraTreesClassifier', 'ExtraTreesRegressor']
        gbdt_list = ['GradientBoostingClassifier', 'GradientBoostingRegressor']

        if model.__class__.__name__ in tree_list:
            return c.tree
        elif model.__class__.__name__ in forest_list:
            return c.forest
        elif model.__class__.__name__ in ada_list:
            return c.ada
        elif model.__class__.__name__ in extra_list:
            return c.extra
        elif model.__class__.__name__ in gbdt_list:
            return c.gbdt

    def __split(self):
        """Split dataset and target to train/test sets."""
        return train_test_split(
            self.X, self.Y,
            test_size=self.seed,
            random_state=self.rnd_state,
            stratify=self.Y if self.Y_type == c.clf else None)

    def __log(self, text: Any):
        """Print log info if 'debug' is on."""
        if self.debug >= 1:
            print('{:-^50}'.format("-"))
            print('-=[ ', text, ' ]=-')

    def __define_model(self):
        """Define model class to create."""
        if self.model_type == c.tree:
            if self.Y_type == c.clf:
                model = DecisionTreeClassifier()
                if not self.params:
                    params = {**self.tree_clf_criterion, **self.tree_params}
            elif self.Y_type == c.regr:
                model = DecisionTreeRegressor()
                if not self.params:
                    params = {**self.tree_regr_criterion, **self.tree_params}

        elif self.model_type == c.forest:
            if self.Y_type == c.clf:
                model = RandomForestClassifier()
                if not self.params:
                    params = {**self.tree_clf_criterion, **self.tree_params,
                              **self.forest_params}
            elif self.Y_type == c.regr:
                model = RandomForestRegressor()
                if not self.params:
                    params = {**self.forest_regr_criterion, **self.tree_params,
                              **self.forest_params}

        elif self.model_type == c.ada:
            if self.Y_type == c.clf:
                model = AdaBoostClassifier()
                if not self.params:
                    params = {**self.ada_params}
            elif self.Y_type == c.regr:
                model = AdaBoostRegressor()
                if not self.params:
                    params = {**self.ada_regr_criterion, **self.ada_params}

        elif self.model_type == c.extra:
            if self.Y_type == c.clf:
                model = ExtraTreesClassifier()
                if not self.params:
                    params = {**self.tree_clf_criterion, **self.tree_params,
                              **self.forest_params}
            elif self.Y_type == c.regr:
                model = ExtraTreesRegressor()
                if not self.params:
                    params = {**self.extra_regr_criterion, **self.tree_params,
                              **self.forest_params}

        elif self.model_type == c.gbdt:
            if self.Y_type == c.clf:
                model = GradientBoostingClassifier()
                if not self.params:
                    params = {**self.gbdt_clf_criterion, **self.tree_params,
                              **self.forest_params}
            elif self.Y_type == c.regr:
                model = GradientBoostingRegressor()
                if not self.params:
                    params = {**self.gbdt_regr_criterion, **self.tree_params,
                              **self.forest_params}
        return model, params

    def tree_search(self,
                    search_type=c.rand,
                    model_type=c.tree,
                    n_jobs=1,
                    random_n_iter=10,
                    params=None
                    ):
        """Train model.

        Retrive training dataset, target class(variable),
        hyperparams and other from class attributes.

        Best trained model saved as Class attribute (TreePredictor.model).

        Parameters
        ----------
        search_type : {"rand", "grid", "bayes_search", "optuna_search"}, default "rand"
            * if 'rand' use random search to tune hyperparameters.
            * if 'grid' use grid search to tune hyperparameters.
            * if 'bayes_search' use bayes optimization to tune hyperparameters
              as currently drafted, model_type and Y_type ignored,
              uses random forest clf directly
            * if 'optuna_search' use Optuna hyperparameter optimization framework
              as currently drafted, model_type and Y_type ignored,
              uses random forest clf directly,
              https://optuna.org

        model_type : {"tree", "forest", "ada", "extra", "gbdt"}, default "tree"
            * if 'tree' train tree model.
            * if 'forest' train random forest model.
            * if 'ada' train ada boost model.
            * if 'extra' train extra tree model.
            * if 'gbdt' train gradient tree boosting model.

        n_jobs : int, default = 1
            Number of jobs for search hyperparams (-1 to use all cores)

        random_n_iter : int, default = 100
            Number of sampled params setting (used for random search)

        params: dict of hyperparameters
            use defaults if not defined

        Returns
        -------
            Best trained model
        """
        self.search_type = search_type
        self.model_type = model_type
        self.params = params

        start_timer = time.time()
        self.__log(model_type + " " + search_type + " " + self.Y_type)
        self.__log(" start: " + time.strftime("%H:%M:%S", time.gmtime(start_timer)))

        X_train, x_test, y_train, y_test = self.__split()

        model, params = self.__define_model()

        if search_type == c.rand:
            tree_search = RandomizedSearchCV(model,
                                             param_distributions=params,
                                             n_iter=random_n_iter,
                                             cv=5,
                                             verbose=self.debug,
                                             n_jobs=n_jobs)
        elif search_type == c.grid:
            tree_search = GridSearchCV(model,
                                       param_grid=params,
                                       cv=5,
                                       verbose=self.debug,
                                       n_jobs=n_jobs)

        elif search_type == c.bayes_search:
            model = RandomForestClassifier()
            tree_search = BayesSearchCV(model,
                                        search_spaces=self.bayes_search_space,
                                        cv=5,
                                        verbose=self.debug,
                                        n_jobs=n_jobs,
                                        n_iter=random_n_iter)

        elif search_type == c.optuna_search:
            params = {**self.tree_clf_criterion, **self.tree_params,
                      **self.forest_params}

            def objective(trial):
                n_estimators = trial.suggest_categorical('n_estimators', params['n_estimators'])
                criterion = trial.suggest_categorical('criterion', params['criterion'])
                max_depth = trial.suggest_int('max_depth', params['max_depth'].start,
                                              params['max_depth'].stop)
                min_samples_split = trial.suggest_int('min_samples_split', params['min_samples_split'].start,
                                                      params['min_samples_split'].stop)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', params['min_samples_leaf'].start,
                                                     params['min_samples_leaf'].stop)
                max_features = trial.suggest_int('max_features', params['max_features'].start,
                                                 params['max_features'].stop)
                max_leaf_nodes = trial.suggest_int('max_leaf_nodes', params['max_leaf_nodes'].start,
                                                   params['max_leaf_nodes'].stop)

                search_model = RandomForestClassifier(n_estimators=n_estimators,
                                               criterion=criterion,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf,
                                               max_features=max_features,
                                               max_leaf_nodes=max_leaf_nodes,
                                               n_jobs=n_jobs,
                                               verbose=self.debug)
                search_model.fit(X_train, y_train.values.ravel())
                return cross_val_score(model, X_train, y_train, cv=5).mean()

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=random_n_iter)

            elapsed_time = time.time() - start_timer
            self.__log('finished: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
            self.__log(study.best_value)
            self.__log(study.best_trial)

            params = study.best_params
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                max_leaf_nodes=params['max_leaf_nodes'],
                n_jobs=n_jobs,
                verbose=self.debug)
            model.fit(X_train, y_train.values.ravel())
            self.trained_model = model
            self.__log("Class " + model.__class__.__name__)
            self.__log('{:-^50}'.format("-"))

            return model

        self.__log("Class " + model.__class__.__name__)

        tree_search.fit(X_train, y_train.values.ravel())
        self.trained_model = tree_search.best_estimator_

        elapsed_time = time.time() - start_timer
        self.__log('finished: ' + time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        self.__log('{:-^50}'.format("-"))

        return tree_search.best_estimator_

    def print_metrics(self):
        """Print params and metrics from last model training.

        Training dataset, target class (variable)
        and model retrieved from class private attributes,

        """

        def prnt(col1: Any, col2: Any):
            # simple table-like print out
            print('{: <35}'.format(col1), '|', col2)
            print('{:-^50}'.format("-"))

        model = self.trained_model
        X_train, x_test, y_train, y_test = self.__split()
        y_pred = model.predict(x_test)

        print('{:-^50}'.format("-"))
        prnt("Class", model.__class__.__name__)
        print("Params:", model.get_params())
        print('{:-^50}'.format("-"))

        if self.Y_type == c.clf:
            # y_pred_proba = model.predict_proba(x_test)
            prnt("Accuracy", accuracy_score(y_test, y_pred))
            prnt("Precision", precision_score(y_test, y_pred))
            prnt("Recall", recall_score(y_test, y_pred))
            prnt("F1-Score", f1_score(y_test, y_pred))
            prnt("Score", model.score(self.X, self.Y))
            cv_scores = cross_val_score(model, X_train,
                                        y_train.values.ravel(),
                                        cv=5, scoring='accuracy')
            prnt("CV Scores", cv_scores)
            prnt("AvgCV Score", cv_scores.mean())
            print(classification_report(y_test, y_pred))

        elif self.Y_type == c.regr:
            prnt("Score", model.score(self.X, self.Y))
            prnt("R squared", r2_score(y_test, y_pred))
            prnt("Mean absolute error", mean_absolute_error(y_test,
                                                            y_pred))
            prnt("Mean absolute error", median_absolute_error(y_test,
                                                              y_pred))
            prnt("Mean squared error", mean_squared_error(y_test, y_pred))

            prnt("Mean absolute percentage err",
                 mean_absolute_percentage_error(y_test, y_pred))
        print('{:-^50}'.format("-"))

    def draw_model(self, w=0, h=0):
        """Draw classifier. Supports tree and forest.

        Does not make sense on real models.

        w,h: int, size of figure
            calculated inside by a simplified rule
        """
        model = self.trained_model
        model_type = self.__get_model_type()

        X_train, x_test, y_train, y_test = self.__split()

        if model_type == c.tree:
            depth = model.get_params()['max_depth']
            leafes = model.get_params()['max_leaf_nodes']
            feats = model.get_params()['max_features']
            # dumb logic, but no sense of improvement.
            h = depth * 2 if h == 0 else h
            h = 65536 if h > 65536 else h
            w = leafes * feats * 6 / h if w == 0 else w
            w = 65536 if w > 65536 else w

            fig, ax = plt.subplots(figsize=(w, h), dpi=300)
            plot_tree(model, feature_names=list(self.X),
                      class_names=['loss', 'profit'],
                      filled=True, ax=ax, fontsize=8)
            fig.savefig("graph/model_tree.png")

        elif model_type in [c.forest, c.ada, c.extra]:
            if model_type in [c.forest, c.extra]:
                h = 50 if h == 0 else h
                w = 50 if w == 0 else w
            elif model_type == c.ada:
                h = 20 if h == 0 else h
                w = 20 if w == 0 else w
            fig, ax = plt.subplots(nrows=model.get_params()['n_estimators'],
                                   ncols=1, figsize=(h, w), dpi=500)
            for i in range(0, len(model.estimators_)):
                plot_tree(model.estimators_[i], feature_names=list(self.X),
                          class_names=['loss', 'profit'],
                          filled=True, ax=ax[i], fontsize=8)
                ax[i].set_title('Estimator: ' + str(i), fontsize=8)
            fig.savefig("graph/model_" + model_type + ".png")

    def draw_metrics(self):
        """Draw metrics from last model training.

        Classification mainly - predict_proba, roc, ...
        """
        model = self.trained_model

        X_train, x_test, y_train, y_test = self.__split()
        # ----------------------------
        fig_importance = plt.figure(figsize=(30, 30))
        plt.title("Feature Importance")
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx],
                 align='center')
        plt.yticks(range(len(sorted_idx)),
                   np.array(x_test.columns)[sorted_idx])
        fig_importance.savefig("graph/feat_importance.png")
        # ----------------------------
        if self.Y_type == c.clf:
            y_pred_proba = model.predict_proba(x_test)
            fig, axes = plt.subplots(nrows=1, figsize=(6, 6))
            plt.title("PREDICT_PROBABILITY")
            plt.hist(y_pred_proba, color=['green', 'orange'])
            fig.savefig("graph/feat_pred_proba.png")
            # ----------------------------
            fig, axes = plt.subplots(ncols=2, figsize=(18, 5))
            axes[0].axhline(0.9, c='b', ls="--", lw=2, alpha=0.5)
            axes[0].axvline(0.9, c='b', ls="--", lw=2, alpha=0.5)
            axes[0].set_title("PR Curve")
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.0])
            plot_precision_recall_curve(model, x_test, y_test, ax=axes[0],
                                        color='green', lw=3)
            axes[1].axhline(0.9, c='b', ls="--", lw=2, alpha=0.5)
            axes[1].axvline(0.1, c='b', ls="--", lw=2, alpha=0.5)
            axes[1].set_title("ROC Curve")
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.0])
            plot_roc_curve(model, x_test, y_test, ax=axes[1],
                           color='darkorange', lw=3)
            axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
            fig.tight_layout()
            fig.savefig("graph/feat_PR_ROC.png")
            # ----------------------------
            fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
            plot_confusion_matrix(model, x_test, y_test,
                                  ax=axes, cmap='plasma')
            fig.tight_layout()
            fig.savefig("graph/feat_confusion.png")

        elif self.Y_type == c.regr:
            pass
