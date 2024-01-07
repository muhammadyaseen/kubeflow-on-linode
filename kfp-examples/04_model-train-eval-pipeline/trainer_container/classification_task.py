from typing import List, Union, Tuple

from sklearn.model_selection import (train_test_split, cross_validate,
                                     GridSearchCV)
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             precision_recall_curve, roc_curve, balanced_accuracy_score)
import pandas as pd
import matplotlib.pyplot as plt



class Task:

    def __init__(
            self,
            name: str,
            model,
            model_params,
            hparams,
            scoring_criteria: Tuple[str],
            refit_criterion: str
    ):

        self.name = name
        self.hparams = hparams
        self.model = model(**model_params)
        self.model_params = model_params
        self.refit_criterion = refit_criterion
        self.scoring_criteria = scoring_criteria

        self.task_model_with_grid_search_trainer = None

    def train(self, X_train, Y_train, verbose=2, cv_folds=5):

        # we can do a CV search over the passed in params
        # This automatically takes care of Stratified K-Folds
        self.task_model_with_grid_search_trainer = GridSearchCV(
            self.model,
            self.hparams,
            scoring=self.scoring_criteria,
            refit=self.refit_criterion,
            cv=cv_folds,
            verbose=verbose,
            n_jobs=-1 # we loose the verbose logs if this is enabled, but the computation time is almosted halved.
        )

        self.task_model_with_grid_search_trainer.fit(
            X_train,
            Y_train
        )


    def test(self, X_test, X_train=None):

        if self.task_model_with_grid_search_trainer is None:
            raise ValueError("Please train the models first using `train` function.")

        Y_test_predicted = self.task_model_with_grid_search_trainer.\
            best_estimator_.\
            predict(X_test)

        if X_train is not None:
            Y_train_predicted = self.task_model_with_grid_search_trainer.\
            best_estimator_.\
            predict(X_train)

        else:
            Y_train_predicted = None

        return Y_test_predicted, Y_train_predicted

    def show_best_model(self,):

        if self.task_model_with_grid_search_trainer is None:
            raise ValueError("Please train the models first using `train` function.")

        base_metric = f'rank_test_{self.refit_criterion}'
        extra_metrics = [
            f'mean_test_{self.refit_criterion}',
            f'rank_test_{self.scoring_criteria[1]}',
            f'mean_test_{self.scoring_criteria[1]}'
        ]

        cv_and_gs_outcomes = pd.DataFrame(self.task_model_with_grid_search_trainer.cv_results_)
        top_ranked_models = cv_and_gs_outcomes[cv_and_gs_outcomes[base_metric] == 1]

        return top_ranked_models[[base_metric] + extra_metrics + ['params']]


class ClassificationTask(Task):

    def __init__(
            self,
            name: str,
            model,
            model_params,
            hparams,
            scoring_criteria: Tuple[str] = ("balanced_accuracy", "accuracy"),
            refit_criterion: str = "balanced_accuracy"
    ) -> None:

        super().__init__(
            name,
            model,
            model_params,
            hparams,
            scoring_criteria,
            refit_criterion,
        )

        self.task_type = "classification"

    def plot_confusion_matrix(
            self,
            y_train,
            y_train_predicted,
            y_test,
            y_test_predicted,
            normalize='true'
    ):


        fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_train,
            y_train_predicted,
            ax=axes[0],
            normalize=normalize,
            colorbar=False
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_test_predicted,
            ax=axes[1],
            normalize=normalize,
            colorbar=False
        )

        return fig, axes

    def plot_roc_and_pr_curves(self, y_true, y_scores):

        precision, recall, fpr, tpr = {},{},{}, {}
        classes = ['Basic', 'Plus', 'Premium']
        fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

        for i, c in enumerate(classes):

            precision[i], recall[i], thresholds = precision_recall_curve(
                y_true == c,
                y_scores[:, i]
            )
            axes[0].plot(
                recall[i],
                precision[i],
                label=c
            )

            fpr[i], tpr[i], _ = roc_curve(y_true == c, y_scores[:, i])
            axes[1].plot(
                fpr[i],
                tpr[i],
                label=c
            )

        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].legend(loc="best")
        axes[0].set_title("Precision vs. Recall curve (OVR)")

        axes[1].set_xlabel("FPR")
        axes[1].set_ylabel("TPR")
        axes[1].legend(loc="best")
        axes[1].set_title("ROC curve")

        return fig, axes
