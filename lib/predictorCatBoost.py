import optuna
import numpy as np
import lib.constants as c
import scipy.stats as stats
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
                             matthews_corrcoef, explained_variance_score,
                             max_error)
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
            # "objective": trial.suggest_categorical("objective",
            # ["Logloss", "CrossEntropy"]),
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
                                                                ["Logloss", "CrossEntropy"])

            model = CatBoostClassifier(**params,
                                       eval_metric="F1",
                                       custom_metric=["F1", "AUC", "Accuracy"],
                                       task_type=c.task_type)
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
                                                                ["RMSE", "MAE"])

            model = CatBoostRegressor(**params,
                                      eval_metric="MAPE",
                                      custom_metric=["R2", "MAE", "RMSE"],
                                      task_type=c.task_type)

            cv_results = cv(Pool(X_train, y_train),
                            model.get_params(),
                            verbose_eval=False,
                            early_stopping_rounds=100
                            )

            self.__log("RMSE from CV")
            self.__log(np.max(cv_results['test-RMSE-mean']), False)
            self.__log(np.max(cv_results['test-RMSE-std']), False)
            self.__log("R2", False)
            self.__log(np.max(cv_results['test-R2-mean']), False)
            self.__log(np.max(cv_results['test-R2-std']), False)
            self.__log("MAE", False)
            self.__log(np.max(cv_results['test-MAE-mean']), False)
            self.__log(np.max(cv_results['test-MAE-std']), False)
            self.__log("MAPE", False)
            self.__log(np.max(cv_results['test-MAPE-mean']), False)
            self.__log(np.max(cv_results['test-MAPE-std']), False)

            return cv_results['test-MAPE-mean'].min()

        else:
            print("Wrong Y_type")
            return

        # !!! partial fit
        # !!! pruner exception
        # !!! may be change cv to handmade loop throu fit`s

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

        # show_progress_bar = True if c.debug >= 1 else False
        show_progress_bar = False
        self.study.optimize(self.__objective,
                            n_trials=c.n_trials,
                            timeout=c.timeout,
                            n_jobs=c.n_jobs,
                            gc_after_trial=c.gc_after_trial,
                            show_progress_bar=show_progress_bar)

        elapsed_time = time.time() - start_timer
        self.__log('finished: ' + time.strftime("%H:%M:%S",
                   time.gmtime(elapsed_time)))

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
            # Вероятности положительного класса
            y_proba = self.model.predict_proba(self.X)[:, 1]
            print("ROC AUC: ", roc_auc_score(self.Y, y_proba))
            print("Log Loss: ", log_loss(self.Y, y_proba))
            print("Cohen's Kappa: ", cohen_kappa_score(self.Y, y_pred))
            print("Matthews Correlation Coefficient MCC: ",
                  matthews_corrcoef(self.Y, y_pred))
            print("Score: ", self.model.score(self.X, self.Y))
            print("")
            print(classification_report(self.Y, y_pred))
        else:
            print("MSE: ", mean_squared_error(self.Y, y_pred))
            print("MAE: ", mean_absolute_error(self.Y, y_pred))
            print("r2: ", r2_score(self.Y, y_pred))
            print("MAPE: ", mean_absolute_percentage_error(self.Y, y_pred))
            print("MedAE: ", median_absolute_error(self.Y, y_pred))
            print("Explained Variance Score: ",
                  explained_variance_score(self.Y, y_pred))
            print("MaxError: ", max_error(self.Y, y_pred))
            cv = np.std(self.Y - y_pred) / np.mean(self.Y - y_pred) * 100
            print("Coeff of Variation: ", cv)
            n = len(self.Y)
            adjusted_r2 = 1 - (1 - r2_score(self.Y, y_pred)
                               ) * (n - 1) / (n - 1 - 1)
            print("Adjusted R2: ", adjusted_r2)

    def draw_metrics(self):
        """."""
        # !!! look at
        # https://github.com/shap/shap
        # https://github.com/catboost/tutorials/blob/master/model_analysis/shap_values_tutorial.ipynb
        # https://catboost.ai/docs/en/concepts/python-reference_catboost_calc_feature_statistics

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
            cm_display.figure_.savefig(
                c.graph_folder + self.name + "feat_confusion.png")
            self.__log('save confusion to: ' + c.graph_folder)

            # Receiver Operating Characteristic
            roc_display = RocCurveDisplay.from_estimator(
                self.model, self.X, self.Y)
            roc_display.figure_.savefig(
                c.graph_folder + self.name + "feat_ROC.png")
            self.__log('save ROC to: ' + c.graph_folder)

            # Precision Recall
            pr_display = PrecisionRecallDisplay.from_estimator(
                self.model, self.X, self.Y)
            pr_display.figure_.savefig(
                c.graph_folder + self.name + "feat_PR.png")
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
            #  pred vs true
            plt.figure(figsize=(100, 100))
            plt.scatter(self.Y, y_pred)
            plt.plot([self.Y.min(), self.Y.max()], [
                     self.Y.min(), self.Y.max()], 'k--', lw=2)
            plt.xlabel('Фактические значения')
            plt.ylabel('Предсказанные значения')
            plt.title('Предсказанные значения против фактических')
            plt.savefig(c.graph_folder + self.name + "pred_true.png")
            self.__log('save pred vs true to: ' + c.graph_folder)
            plt.close()
            #  residuals
            plt.figure(figsize=(100, 100))
            residuals = self.Y - y_pred
            plt.scatter(y_pred, residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Предсказанные значения')
            plt.ylabel('Остатки')
            plt.title('График остатков')
            plt.savefig(c.graph_folder + self.name + "residuals.png")
            self.__log('save resuduals to: ' + c.graph_folder)
            plt.close()
            #  residuals hist
            plt.figure(figsize=(100, 100))
            plt.hist(residuals, bins=30, edgecolor='k')
            plt.xlabel('Остатки')
            plt.ylabel('Частота')
            plt.title('Гистограмма остатков')
            plt.savefig(c.graph_folder + self.name + "residuals_hist.png")
            self.__log('save resuduals hist to: ' + c.graph_folder)
            plt.close()
            #  Quantile-Quantile
            plt.figure(figsize=(100, 100))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q график остатков')
            plt.savefig(c.graph_folder + self.name + "Q_Q.png")
            self.__log('save quantile-quantile to: ' + c.graph_folder)
            plt.close()

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
