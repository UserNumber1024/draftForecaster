import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
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
from sklearn.tree import (DecisionTreeClassifier,
                          DecisionTreeRegressor, plot_tree)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split, cross_val_score)
# CONSTANTS--------------------------------------------------------------------
tree = 'tree'
extra = 'extra'
forest = 'forest'
gbdt = 'gbdt'
ada = 'ada'
clf = 'clf'
regr = 'regr'
rand = 'rand'
grid = 'grid'
# -----------------------------------------------------------------------------


class TreePredictor:
    """Tree-like models training."""

    def __init__(self, X: pd.DataFrame, Y: pd.DataFrame, Y_type: str,
                 rnd_state=0, seed=0.33,
                 debug=0, ):
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
        self.X = X
        self.Y = Y
        self.Y_type = Y_type
        self.debug = debug
        self.tree_clf_criterion = {'criterion': ['gini', 'entropy']}
        self.tree_regr_criterion = {'criterion': [
            "squared_error", "friedman_mse", "absolute_error", "poisson"]}
        self.tree_params = {'max_depth': range(5, 100),
                            'max_features': range(4, 60),
                            'min_samples_split': range(2, 10),
                            'min_samples_leaf': range(2, 5),
                            'max_leaf_nodes': range(10, 400)}
        self.forest_params = {'n_estimators': [7, 5, 3]  # 20, 50, 100, 150],
                              }
        self.forest_regr_criterion = {'criterion': [
            "squared_error", "absolute_error", "poisson"]}
        self.ada_params = {'n_estimators': [10, 7, 5, 3, 12],
                           # 'n_estimators': [10, 5, 20, 50, 100, 150, 200],
                           "learning_rate": [0.1, 0.01, 0.2, 0.05, 0.5, 1]
                           }
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

    def __get_model_type(self):
        """Define model_type by model from Class attributes."""
        model = self.trained_model

        tree_list = ['DecisionTreeClassifier', 'DecisionTreeRegressor']
        forest_list = ['RandomForestClassifier', 'RandomForestRegressor']
        ada_list = ['AdaBoostClassifier', 'AdaBoostRegressor']
        extra_list = ['ExtraTreesClassifier', 'ExtraTreesRegressor']
        gbdt_list = ['GradientBoostingClassifier', 'GradientBoostingRegressor']

        if model.__class__.__name__ in tree_list:
            return tree
        elif model.__class__.__name__ in forest_list:
            return forest
        elif model.__class__.__name__ in ada_list:
            return ada
        elif model.__class__.__name__ in extra_list:
            return extra
        elif model.__class__.__name__ in gbdt_list:
            return gbdt

    def __split(self):
        """Split dataset and target to train/test sets."""
        return train_test_split(
            self.X, self.Y,
            test_size=self.seed,
            random_state=self.rnd_state,
            stratify=self.Y if self.Y_type == clf else None)

    def tree_search(self,
                    search_type=rand,
                    model_type='tree',
                    n_jobs=1,
                    random_n_iter=100,
                    params={}
                    ):
        """Train model.

        Retrive training dataset, target class(variable),
        hyperparams and other from class attributes.

        Best trained model saved as Class attribute (TreePredictor.model).

        Parameters
        ----------
        search_type : {"rand", "grid"}
            * if 'rand' use random search to tune hyperparameters.
            * if 'grid' use grid search to tune hyperparameters.

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

        Returns
        -------
            Best trained model
        """
        if self.debug >= 1:
            start_timer = time.time()
            print('{:-^50}'.format("-"))
            print("-=[ ", model_type, " ", search_type, " ", self.Y_type,
                  " start: ",
                  time.strftime("%H:%M:%S", time.gmtime(start_timer)), " ]=-")

        X_train, x_test, y_train, y_test = self.__split()

        if model_type == tree:
            if self.Y_type == clf:
                model = DecisionTreeClassifier()
                if params == {}:
                    params = {**self.tree_clf_criterion, **self.tree_params}
            elif self.Y_type == regr:
                model = DecisionTreeRegressor()
                if params == {}:
                    params = {**self.tree_regr_criterion, **self.tree_params}

        elif model_type == forest:
            if self.Y_type == clf:
                model = RandomForestClassifier()
                if params == {}:
                    params = {**self.tree_clf_criterion, **self.tree_params,
                              **self.forest_params}
            elif self.Y_type == regr:
                model = RandomForestRegressor()
                if params == {}:
                    params = {**self.forest_regr_criterion, **self.tree_params,
                              **self.forest_params}

        elif model_type == ada:
            if self.Y_type == clf:
                model = AdaBoostClassifier()
                if params == {}:
                    params = {**self.ada_params}
            elif self.Y_type == regr:
                model = AdaBoostRegressor()
                if params == {}:
                    params = {**self.ada_regr_criterion, **self.ada_params}

        elif model_type == extra:
            if self.Y_type == clf:
                model = ExtraTreesClassifier()
                if params == {}:
                    params = {**self.tree_clf_criterion, **self.tree_params,
                              **self.forest_params}
            elif self.Y_type == regr:
                model = ExtraTreesRegressor()
                if params == {}:
                    params = {**self.extra_regr_criterion, **self.tree_params,
                              **self.forest_params}

        elif model_type == gbdt:
            if self.Y_type == clf:
                model = GradientBoostingClassifier()
                if params == {}:
                    params = {**self.gbdt_clf_criterion, **self.tree_params,
                              **self.forest_params}
            elif self.Y_type == regr:
                model = GradientBoostingRegressor()
                if params == {}:
                    params = {**self.gbdt_regr_criterion, **self.tree_params,
                              **self.forest_params}

        if search_type == rand:
            tree_search = RandomizedSearchCV(model,
                                             param_distributions=params,
                                             n_iter=random_n_iter,
                                             cv=5,
                                             verbose=self.debug,
                                             n_jobs=n_jobs)
        elif search_type == grid:
            tree_search = GridSearchCV(model,
                                       param_grid=params,
                                       cv=5,
                                       verbose=self.debug,
                                       n_jobs=n_jobs)

        if self.debug >= 1:
            print("Class ", model.__class__.__name__)

        tree_search.fit(X_train, y_train.values.ravel())
        model = model.fit(X_train, y_train.values.ravel())

        self.trained_model = tree_search.best_estimator_

        if self.debug >= 1:
            elapsed_time = time.time() - start_timer
            print("-=[ finished: ",
                  time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), "]=-")
            print('{:-^50}'.format("-"))

        return tree_search.best_estimator_

    def print_metrics(self):
        """Print params and metrics from last model training.

        Training dataset, target class (variable)
        and model retrieved from class private attributes,

        """
        def prnt(col1='', col2=''):
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

        if self.Y_type == clf:
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

        elif self.Y_type == regr:
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
            calculated inside by a simlified rule
        """
        model = self.trained_model
        model_type = self.__get_model_type()

        X_train, x_test, y_train, y_test = self.__split()

        if model_type == tree:
            depth = model.get_params()['max_depth']
            leafes = model.get_params()['max_leaf_nodes']
            feats = model.get_params()['max_features']
            # dumb logic, but no sense of improvement.
            h = depth * 2 if h == 0 else h
            w = leafes * feats * 6 / h if w == 0 else w

            fig, ax = plt.subplots(figsize=(w, h), dpi=300)
            plot_tree(model, feature_names=list(self.X),
                      class_names=['loss', 'profit'],
                      filled=True, ax=ax, fontsize=8)
            plt.show()
            fig.savefig("graph/model_tree.png")

        elif model_type in [forest, ada, extra]:
            if model_type in [forest, extra]:
                h = 50 if h == 0 else h
                w = 50 if w == 0 else w
            elif (model_type == ada):
                h = 20 if h == 0 else h
                w = 20 if w == 0 else w
            fig, ax = plt.subplots(nrows=model.get_params()['n_estimators'],
                                   ncols=1, figsize=(h, w), dpi=500)
            for i in range(0, len(model.estimators_)):
                plot_tree(model.estimators_[i], feature_names=list(self.X),
                          class_names=['loss', 'profit'],
                          filled=True, ax=ax[i], fontsize=8)
                ax[i].set_title('Estimator: ' + str(i), fontsize=8)
            plt.show()
            fig.savefig("graph/model_"+model_type+".png")

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
        if self.Y_type == clf:
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

        elif self.Y_type == regr:
            pass
