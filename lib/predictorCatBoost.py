import optuna
import numpy as np
import lib.constants as c
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error,
                             mean_absolute_percentage_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             roc_auc_score, make_scorer,
                             ConfusionMatrixDisplay, confusion_matrix,
                             RocCurveDisplay, roc_curve,
                             PrecisionRecallDisplay, precision_recall_curve)
import matplotlib.pyplot as plt
import joblib
from typing import Any
import time
import seaborn as sns


class ModelCatBoost:
    """Catboost models training.

    https://catboost.ai/docs/en/
    """

    def __init__(self, X, Y, Y_type=c.clf, rnd_state=0):
        self.X = X
        self.Y = Y
        self.Y_type = Y_type
        self.model = None
        self.study = None
        self.debug = c.debug
        # self.seed = seed    # !!!!!!!
        self.rnd_state = np.random.randint(
            1, 500) if rnd_state == 0 else rnd_state    # !!!!!!!

    def __log(self, text: Any, divider=True):
        """Print log info if 'debug' is on."""
        if self.debug >= 1:
            if divider:
                print('{:-^50}'.format("-"))
            print('-=[ ', text, ' ]=-')

    def __objective(self, trial):
        """."""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
        }

        if self.Y_type == c.clf:
            model = CatBoostClassifier(**params)
        elif self.Y_type == c.clf:
            model = CatBoostRegressor(**params)
        else:
            print("Wrong Y_type")
            return

        # Используем Pool для CatBoost
        # какие ещё вариант и что такое пул
        pool = Pool(self.X, self.Y)
        # !!! partial fit
        # !!! pruner exception
        # !!! supress 319:	learn: 0.3649491	total: 29.8s	remaining: 186ms
        model.fit(pool, verbose_eval=False)

        if self.Y_type == c.clf:
            y_pred = model.predict(self.X)
            # scorer = make_scorer(roc_auc_score, needs_proba=True)
            # scores = cross_val_score(
            # search_model, X_train, y_train, scoring=scorer, cv=kf)
            # !!! что-то другое, не аккураси
            return accuracy_score(self.Y, y_pred)
        else:
            y_pred = model.predict(self.X)
            return -mean_squared_error(self.Y, y_pred)  # Минимизируем MSE

    def optimize_params(self):
        """."""
        if self.debug >= 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        start_timer = time.time()
        self.__log(self.Y_type)
        self.__log(" start: " + time.strftime("%H:%M:%S",
                   time.gmtime(start_timer)), False)

        direction = 'maximize' if self.Y_type == c.clf else 'minimize'
        pruner = optuna.pruners.MedianPruner()
        # !!! прунер подобрать

        self.study = optuna.create_study(pruner=pruner,
                                         direction=direction)

        show_progress_bar = True if c.debug >= 1 else False
        self.study.optimize(self.__objective,
                            n_trials=c.n_trials,
                            timeout=None,
                            n_jobs=c.n_jobs,
                            gc_after_trial=False,
                            show_progress_bar=show_progress_bar)

        elapsed_time = time.time() - start_timer
        self.__log('finished: ' + time.strftime("%H:%M:%S",
                   time.gmtime(elapsed_time)), False)

    def print_best_params(self):
        """."""
        if self.study is None:
            print("No study.")
            return

        print('{:-^50}'.format("-"))
        print("-=[ Best hyperparams ]=-")
        for key, value in self.study.best_params.items():
            print(f"{key}: {value}")

        print('{:-^50}'.format("-"))
        importances = optuna.importance.get_param_importances(self.study)
        print("-=[ Hyperparameter Importances ]=-")
        for param, importance in importances.items():
            print(f"{param}: {importance:.4f}")

    def print_metrics(self):
        """."""
        if self.model is None:
            print("Model is not trained.")
            return

        print('{:-^50}'.format("-"))
        y_pred = self.model.predict(self.X)
        if self.Y_type == c.clf:
            accuracy = accuracy_score(self.Y, y_pred)
            precision = precision_score(self.Y, y_pred, average='weighted')
            recall = recall_score(self.Y, y_pred, average='weighted')
            print(
                f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(classification_report(
                self.Y, y_pred))
        else:
            mse = mean_squared_error(self.Y, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")

    def draw_metrics(self):
        """."""
        if self.study is None:
            print("No study.")
            return

        y_pred = self.model.predict(self.X)

        # Feature Importance
        feature_importances = self.model.get_feature_importance()
        feature_names = self.X.columns if hasattr(
            self.X, 'columns') else np.arange(self.X.shape[1])
        sorted_indices = np.argsort(feature_importances)
        plt.figure(figsize=(20, 100))
        plt.barh(range(len(feature_importances)),
                 feature_importances[sorted_indices], align='center')
        plt.yticks(range(len(feature_importances)),
                   feature_names[sorted_indices])
        plt.title("Feature Importance")
        plt.savefig(c.graph_folder + "feat_importance.png")
        self.__log('save feat_importance to: ' + c.graph_folder)
        plt.show()
        plt.close()

        if self.Y_type == c.clf:
            y_pred_proba = self.model.predict_proba(self.X)

            # Confusion matrix
            cm_display = ConfusionMatrixDisplay.from_estimator(
                self.model, self.X, self.Y).plot()
            cm_display.figure_.savefig(c.graph_folder + "feat_confusion.png")
            self.__log('save confusion to: ' + c.graph_folder)

            # Receiver Operating Characteristic
            roc_display = RocCurveDisplay.from_estimator(
                self.model, self.X, self.Y).plot()
            roc_display.figure_.savefig(c.graph_folder + "feat_ROC.png")
            self.__log('save ROC to: ' + c.graph_folder)

            # Precision Recall
            pr_display = PrecisionRecallDisplay.from_estimator(
                self.model, self.X, self.Y).plot()
            pr_display.figure_.savefig(c.graph_folder + "feat_PR.png")
            self.__log('save PR to: ' + c.graph_folder)

            #
            fig, axes = plt.subplots(nrows=1, figsize=(25, 25))
            plt.title("PREDICT PROBABILITY")
            plt.hist(y_pred_proba, color=['green', 'orange'])
            fig.savefig(c.graph_folder + "feat_pred_proba.png")
            self.__log('save pred_proba to: ' + c.graph_folder)

        else:
            mse = mean_squared_error(self.Y, y_pred)
            plt.bar(['Mean Squared Error'], [mse])
            plt.title("Model Mean Squared Error")
            plt.ylabel("Score")
            plt.show()
            print(f"Mean Squared Error: {mse:.4f}")


    def export_model(self, filename):
        """."""
        if self.model is None:
            print("Model is not.")
            return
        joblib.dump(self.model, filename)
        print(f"Model exported to {filename}")

    def fit(self):
        """."""
        if self.study is None:
            print("No study.")
            return

        best_params = self.study.best_params
        if self.Y_type == c.clf:
            self.model = CatBoostClassifier(**best_params)
        else:
            self.model = CatBoostRegressor(**best_params)

        pool = Pool(self.X, self.Y)
        self.model.fit(pool, verbose_eval=False)

        self.__log("Class " + self.model.__class__.__name__, False)
