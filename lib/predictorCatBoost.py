import optuna
import numpy as np
import lib.constants as c
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, metrics, cv
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error,
                             mean_absolute_percentage_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report,
                             roc_auc_score, make_scorer,
                             ConfusionMatrixDisplay, confusion_matrix,
                             RocCurveDisplay, roc_curve,
                             PrecisionRecallDisplay, precision_recall_curve,
                             roc_auc_score, log_loss, cohen_kappa_score,
                             matthews_corrcoef)
from sklearn.model_selection import (train_test_split, cross_val_score, KFold)
import matplotlib.pyplot as plt
import joblib
from typing import Any
import time
import seaborn as sns


class ModelCatBoost:
    """Catboost models training.

    https://catboost.ai/docs/en/
    https://github.com/catboost/tutorials

    """

    def __init__(self, X, Y, Y_type=c.clf, rnd_state=0, name='clf_catbst_1'):
        self.X = X
        self.Y = Y
        self.Y_type = Y_type
        self.model = None
        self.study = None
        self.debug = c.debug
        self.seed = c.seed
        self.rnd_state = np.random.randint(
            1, 500) if rnd_state == 0 else rnd_state
        self.name = name

    def __log(self, text: Any, divider=True):
        """Print log info if 'debug' is on."""
        if self.debug >= 1:
            if divider:
                print('{:-^50}'.format("-"))
            print('-=[ ', text, ' ]=-')

    def __split(self):
        """Split dataset and target to train/test sets."""
        self.__log('Split dataset with random_state:')
        self.__log(self.rnd_state, False)
        return train_test_split(
            self.X, self.Y,
            test_size=self.seed,
            random_state=self.rnd_state,
            stratify=self.Y if self.Y_type == c.clf else None)

    def __objective(self, trial):
        """."""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'boosting_type': trial.suggest_categorical('boosting_type',
                                                       ['Ordered', 'Plain']),
            'max_ctr_complexity': trial.suggest_int('max_ctr_complexity', 0, 8),
            "objective": trial.suggest_categorical("objective",
                                                   ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel",
                                                     0.01, 0.1),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type",
                                                        ["Bayesian",
                                                         "Bernoulli", "MVS"])
        }

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        params.update({'random_state': self.rnd_state})
        X_train, _, y_train, _ = self.__split()

        if self.Y_type == c.clf:
            params["loss_function"] = trial.suggest_categorical("loss_function",
                                                                ["Logloss"])

            model = CatBoostClassifier(**params,
                                       eval_metric="F1",
                                       custom_metric=["F1", "AUC", "Accuracy"])
            # !!!!!
            # https://github.com/catboost/tutorials/blob/master/cross_validation/cv_tutorial.ipynb
            cv_results = cv(Pool(X_train, y_train),
                            model.get_params(),
                            verbose_eval=False,
                            early_stopping_rounds=100
                            )

            self.__log("F1 from CV")
            self.__log(np.max(cv_results['test-F1-mean']), False)
            self.__log(np.max(cv_results['test-F1-std']), False)
            self.__log("AUC", False)
            self.__log(np.max(cv_results['test-AUC-mean']), False)
            self.__log(np.max(cv_results['test-AUC-std']), False)
            self.__log("Accuracy", False)
            self.__log(np.max(cv_results['test-Accuracy-mean']), False)
            self.__log(np.max(cv_results['test-Accuracy-std']), False)

            return cv_results['test-F1-mean'].max()
            # !!!! возможно добавить второй параметр, если перекос

        elif self.Y_type == c.regr:
            params["loss_function"] = trial.suggest_categorical("loss_function",
                                                                ["RMSE"])

            model = CatBoostRegressor(**params)
            y_pred = model.predict(self.X)
            return -mean_squared_error(self.Y, y_pred)  # Минимизируем MSE

        else:
            print("Wrong Y_type")
            return

        # !!! partial fit
        # !!! pruner exception

    def optimize_params(self):
        """."""
        if self.debug >= 1:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        start_timer = time.time()
        self.__log(self.Y_type)
        self.__log(self.name)
        self.__log(" start: " + time.strftime("%H:%M:%S",
                   time.gmtime(start_timer)), False)

        direction = 'maximize' if self.Y_type == c.clf else 'minimize'
        pruner = optuna.pruners.MedianPruner()
        # !!! pruner
        # !!! sampler ???

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
        # Вероятности положительного класса
        y_proba = self.model.predict_proba(self.X)[:, 1]
        if self.Y_type == c.clf:
            print("ROC AUC: ", roc_auc_score(self.Y, y_proba))
            print("Log Loss: ", log_loss(self.Y, y_proba))
            print("Cohen's Kappa: ", cohen_kappa_score(self.Y, y_pred))
            print("Matthews Correlation Coefficient MCC: ",
                  matthews_corrcoef(self.Y, y_pred))
            print("Score: ", self.model.score(self.X, self.Y))
            print("")
            print(classification_report(self.Y, y_pred))
        else:
            mse = mean_squared_error(self.Y, y_pred)
            print(f"Mean Squared Error: {mse:.4f}")

    def draw_metrics(self):
        """."""
        # !!! SHAP
        # https://github.com/shap/shap
        # https://github.com/catboost/tutorials/blob/master/model_analysis/shap_values_tutorial.ipynb
        # !!!!!
        # !!! feature statistic
        # https://catboost.ai/docs/en/concepts/python-reference_catboost_calc_feature_statistics
        # https://github.com/catboost/tutorials/blob/master/model_analysis/feature_statistics_tutorial.ipynb

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
        plt.savefig(c.graph_folder + self.name + "feat_imp.png")
        self.__log('save feat_importance to: ' + c.graph_folder)
        plt.close()

        if self.Y_type == c.clf:
            y_pred_proba = self.model.predict_proba(self.X)

            # Confusion matrix
            cm_display = ConfusionMatrixDisplay.from_estimator(
                self.model, self.X, self.Y)
            cm_display.figure_.savefig(c.graph_folder + self.name + "feat_confusion.png")
            self.__log('save confusion to: ' + c.graph_folder)

            # Receiver Operating Characteristic
            roc_display = RocCurveDisplay.from_estimator(
                self.model, self.X, self.Y)
            roc_display.figure_.savefig(c.graph_folder + self.name + "feat_ROC.png")
            self.__log('save ROC to: ' + c.graph_folder)

            # Precision Recall
            pr_display = PrecisionRecallDisplay.from_estimator(
                self.model, self.X, self.Y)
            pr_display.figure_.savefig(c.graph_folder + self.name + "feat_PR.png")
            self.__log('save PR to: ' + c.graph_folder)

            #
            plt.figure(figsize=(25, 25))
            # fig, axes = plt.subplots(nrows=1, figsize=(25, 25))
            plt.title("PREDICT PROBABILITY")
            plt.hist(y_pred_proba, color=['green', 'orange'])
            plt.savefig(c.graph_folder + self.name + "feat_pred_proba.png")
            self.__log('save pred_proba to: ' + c.graph_folder)
            plt.close()

        else:
            mse = mean_squared_error(self.Y, y_pred)
            plt.bar(['Mean Squared Error'], [mse])
            plt.title("Model Mean Squared Error")
            plt.ylabel("Score")
            plt.show()
            print(f"Mean Squared Error: {mse:.4f}")

    def export_model(self):
        """."""
        if self.model is None:
            print("Model is not.")
            return
        filename = c.model_folder + self.name
        joblib.dump(self.model, filename+'.pkl')

        self.model.save_model(filename+'.json',
                              format="json")
        print(f"Model exported to {filename}")

        # to load model from file and show keys
        # import json
        # model = json.load(open("model.json", "r"))
        # model.keys()

    def fit(self):
        """."""
        if self.study is None:
            print("No study.")
            return

        X_train, X_test, y_train, y_test = self.__split()

        best_params = self.study.best_params
        best_params.update({'random_state': self.rnd_state})

        if self.Y_type == c.clf:
            self.model = CatBoostClassifier(**best_params)
        else:
            self.model = CatBoostRegressor(**best_params)

        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_test, y_test)
        self.model.fit(train_pool,
                       eval_set=eval_pool,
                       verbose_eval=False,
                       early_stopping_rounds=100)

        self.__log("Class " + self.model.__class__.__name__, False)
