# !!!!!!!!!!!!!!!!!!!
# Development paused due weak intermediate result
# summary
# - classifier ok, metric ok
# - regressor - not tested (looks like hang up), metric not teted
# - to do - reduce hyperparam range, implement pruner
# pruner https://skine.ru/articles/751791/

import time
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import pandas as pd
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier)
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error,
                             mean_absolute_percentage_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             plot_roc_curve, plot_precision_recall_curve,
                             plot_confusion_matrix,
                             roc_auc_score, make_scorer)
from sklearn.model_selection import (train_test_split, cross_val_score, KFold)
import lib.constants as c
from typing import Any

# -----------------------------------------------------------------------------


class ModelForest:
    """Random forest models training."""

    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, Y_type: str,
                 rnd_state=0, seed=0.33, debug=c.debug):
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
        self.seed = seed
        self.rnd_state = np.random.randint(
            1, 500) if rnd_state == 0 else rnd_state

    def __split(self):
        """Split dataset and target to train/test sets."""
        self.__log('Split dataset with random_state:')
        self.__log(self.rnd_state, False)
        return train_test_split(
            self.X, self.Y,
            test_size=self.seed,
            random_state=self.rnd_state,
            stratify=self.Y if self.Y_type == c.clf else None)

    def __log(self, text: Any, divider=True):
        """Print log info if 'debug' is on."""
        if self.debug >= 1:
            if divider:
                print('{:-^50}'.format("-"))
            print('-=[ ', text, ' ]=-')

    def tree_search(self,
                    n_jobs=1,
                    n_trials=10
                    ):
        """Train model.

        Retrive training dataset, target class(variable),
        hyperparams and other from class attributes.

        Apply Optuna search to RandomForest.
        https://optuna.org

        !!!!!!????Best trained model saved as Class attribute (TreePredictor.model).

        Parameters
        ----------
        n_jobs : int, default = 1
            Number of jobs for search hyperparams (-1 to use all cores)

        n_trials : int, default = 10
            Number of trials for optuna stydy

        params: dict of hyperparameters

        Returns
        -------
            Best trained model
        """
        start_timer = time.time()
        self.__log(self.Y_type)
        self.__log(" start: " + time.strftime("%H:%M:%S",
                   time.gmtime(start_timer)), False)

        X_train, x_test, y_train, y_test = self.__split()

        def objective(t):

            params = {
                'n_estimators': t.suggest_int('n_estimators',
                                              c.n_est.start,
                                              c.n_est.stop),
                'max_depth': t.suggest_int('max_depth',
                                           c.max_dep.start,
                                           c.max_dep.stop),
                'min_samples_split': t.suggest_int('min_samples_split',
                                                   c.min_s_split.start,
                                                   c.min_s_split.stop),
                'min_samples_leaf': t.suggest_int('min_samples_leaf',
                                                  c.min_s_leaf.start,
                                                  c.min_s_leaf.stop),
                'max_features': t.suggest_categorical('max_features',
                                                      c.max_feat),
                'max_leaf_nodes': t.suggest_int('max_leaf_nodes',
                                                c.max_l_nodes.start,
                                                c.max_l_nodes.stop),
                'n_jobs': n_jobs,
                'random_state': self.rnd_state,
                'verbose': self.debug,
                'bootstrap': t.suggest_categorical('bootstrap',
                                                   c.bootstr),
                'warm_start': False
                # !!!
                # min_weight_fraction_leaf
                # min_impurity_decrease
                # ccp_alpha
                # max_samples
            }
            if self.Y_type == c.clf:
                params['criterion'] = t.suggest_categorical('criterion',
                                                            c.f_clf_crit)
                params['class_weight'] = t.suggest_categorical('class_weight',
                                                               c.class_weight)
                search_model = RandomForestClassifier(**params)
                # !!!
                # search_model.fit(X_train, y_train.values.ravel())
                # !!!
                kf = KFold(n_splits=5, shuffle=True,
                           random_state=self.rnd_state)
                scorer = make_scorer(roc_auc_score, needs_proba=True)
                scores = cross_val_score(
                    search_model, X_train, y_train, scoring=scorer, cv=kf)

                # !!! Pruner not really work here
                # need to iterate somehow while fit
                if t.should_prune():
                    raise optuna.TrialPruned()

                return np.min([np.mean(scores), np.median(scores)])

            elif self.Y_type == c.regr:
                params['criterion'] = t.suggest_categorical('criterion',
                                                            c.f_rgr_crit)
                search_model = RandomForestRegressor(**params)

                # !!!
                # search_model.fit(X_train, y_train.values.ravel())
                # !!!
                kf = KFold(n_splits=5, shuffle=True,
                           random_state=self.rnd_state)
                # scorer = make_scorer(neg_mean_squared_error, needs_proba=True)
                # scores = cross_val_score(
                #     search_model, X_train, y_train, scoring=scorer, cv=kf)

                # 5-фолдовая кросс-валидация
                score = cross_val_score(
                    search_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

                # mse_scores = -scores
                self.__log(score)
                return -np.mean(score)

                # score = cross_val_score(model, X_train, y_train, cv=5,
                # scoring='neg_mean_squared_error')
                # return -score.mean()  # Возвращаем среднее значение MSE

        sampler = TPESampler(multivariate=True)

        pruner = HyperbandPruner(min_resource=1,
                                 max_resource=n_trials,
                                 reduction_factor=3)
        direction = 'maximize' if self.Y_type == c.clf else 'minimize'
        study = optuna.create_study(direction=direction,
                                    sampler=sampler,
                                    pruner=pruner,
                                    study_name='maybegivesomenamelater__')

        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs,
                       timeout=None, gc_after_trial=False,
                       show_progress_bar=True)

        
        importances = optuna.importance.get_param_importances(study)
        self.__log("Hyperparameter Importances:", False)
        for param, importance in importances.items():
            self.__log(f"{param}: {importance:.4f}", False)

        elapsed_time = time.time() - start_timer
        self.__log('finished: ' + time.strftime("%H:%M:%S",
                   time.gmtime(elapsed_time)), False)

        if self.Y_type == c.clf:
            model = RandomForestClassifier(**study.best_params)
        elif self.Y_type == c.regr:
            model = RandomForestRegressor(**study.best_params)
        model.fit(X_train, y_train.values.ravel())

        self.trained_model = model
        self.__log("Class " + model.__class__.__name__, False)

        return model

    def print_metrics(self):
        """Print params and metrics from last model training.

        Training dataset, target class (variable)
        and model retrieved from class private attributes,

        """

        def prnt(col1: Any, col2: Any):
            # simple table-like print out
            print('{: <30}'.format(col1), '|', col2)
            print('{:-^50}'.format("-"))

        model = self.trained_model
        X_train, x_test, y_train, y_test = self.__split()
        y_pred = model.predict(x_test)

        print('{:-^50}'.format("-"))
        prnt("Metrics of", model.__class__.__name__)
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
            prnt("R squared (0-bad, 1-good)", r2_score(y_test, y_pred))
            prnt("Mean absolute error", mean_absolute_error(y_test,
                                                            y_pred))
            prnt("Mean absolute error", median_absolute_error(y_test,
                                                              y_pred))
            prnt("Mean squared error", mean_squared_error(y_test, y_pred))

            prnt("Mean absolute percentage err",
                 mean_absolute_percentage_error(y_test, y_pred))
        print('{:-^50}'.format("-"))

    def draw_metrics(self, graph_folder=c.graph_folder):
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
        fig_importance.savefig(graph_folder + "feat_importance.png")
        self.__log('save feat_importance to: ' + graph_folder)
        # ----------------------------
        if self.Y_type == c.clf:
            y_pred_proba = model.predict_proba(x_test)
            fig, axes = plt.subplots(nrows=1, figsize=(6, 6))
            plt.title("PREDICT_PROBABILITY")
            plt.hist(y_pred_proba, color=['green', 'orange'])
            fig.savefig(graph_folder + "feat_pred_proba.png")
            self.__log('save pred_proba to: ' + graph_folder)
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
            fig.savefig(graph_folder + "feat_PR_ROC.png")
            self.__log('save PR_ROC to: ' + graph_folder)
            # ----------------------------
            fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
            plot_confusion_matrix(model, x_test, y_test,
                                  ax=axes, cmap='plasma')
            fig.tight_layout()
            fig.savefig(graph_folder + "feat_confusion.png")
            self.__log('save confusion to: ' + graph_folder)

        elif self.Y_type == c.regr:
            pass
