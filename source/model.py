# General imports
import logging
import os
import numpy as np
import pandas as pd
from pampy import match
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
# Sklearn imports
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV


# Get logger
logger = logging.getLogger("__main__")


def model_factory(model_type: str):
    """
    This function instantiates the correct learner given the specified model type
    """
    return match(model_type,
        'RandomForest',           RandomForestClassifier(criterion='mse'),
        'ExtraTrees',             ExtraTreesClassifier(),
        'XGBoost',                XGBClassifier(),
        'LightGBM',               LGBMClassifier(verbose=-1),
        'CatBoost',               CatBoostClassifier()
        )


class MLmodel(object):

    def __init__(self, settings: dict):

        self.settings = settings
        self.model = model_factory(model_type=settings['model_type'])
        return


    def optimize(self, X, y, params= {}):
        """
        Method that launches the hyperparameter tuning routine.
        I have included in the code the gridsearch method (scan the full hyperparameter space)
        and the random search (sample just some points of the hyperparameter space)
        """

        logger.info('* * * Optimizing hyperparameters')
        tuning_function = self.settings['tuning_function']
        hyperparameter_space = self.settings['hyperparameter_space']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if tuning_function == 'gridsearch':
                logger.info('* * * Launching the Grid Search')
                grid_search_estimator = GridSearchCV(self.model, param_grid=hyperparameter_space, **params)
                grid_search_estimator.fit(X, y)
                self.model = grid_search_estimator.best_estimator_
                logger.info('* * * * Optimal Hyperparameters: {0}'.format(grid_search_estimator.best_params_))
            elif tuning_function  == 'randomsearch':
                logger.info('* * * Launching the Random Search')
                random_search_estimator = RandomizedSearchCV(self.model,param_distributions=hyperparameter_space,**params)
                random_search_estimator.fit(X, y)

                self.model = random_search_estimator.best_estimator_
                logger.info('* * * * Optimal Hyperparameters: {0}'.format(self.model))

            else:
                logger.error("This hyperparameter tuning method doesn't exists!")
                raise NotImplementedError

        logger.info('* * * Hyperparameters optimized!')
        return

    def fit(self, X, y, fit_params={}):
        """
        Method that fit the model
        """
        self.model.fit(X, y, **fit_params)
        return

    def predict(self, X):
        """
        Method that return the prediction of the model
        """
        return self.model.predict_proba(X)[:,1]


    def plotFeaturesImportances(self, train_features):
        """
        A function to plot and print the feature importances of the model.

        """

        tree_model = self.model
        importances = pd.DataFrame({'Importance': tree_model.feature_importances_,
                                    'Feature_name': train_features})

        indices = np.argsort(tree_model.feature_importances_)[::-1]

        importances = importances.sort_values('Importance', ascending=False)
        N_feat = len(importances)

        plt.figure(figsize=(20, 15))
        plt.xticks(range(0, N_feat), importances['Feature_name'], rotation=90)
        plt.xlim([-1, N_feat])
        plt.tight_layout()
        plt.bar(range(0, N_feat),
                        importances['Importance'])
        plt.xlabel('Feature name')
        plt.ylabel('Feature importance')
        plt.grid()
        plt.tight_layout()
        plt.show(block=False)

        logger.info("\n--------- Feature Importance ---------")
        for f in range(0, len(train_features)):
            logger.info("%2d) %-*s %f" % (f + 1, 30, train_features[indices[f]], tree_model.feature_importances_[indices[f]]))
        return

    @staticmethod
    def compute_model_gof_kpis(predictions: pd.DataFrame,
                           true_class_name='true_class',
                           pred_class_name='predicted_class',
                           pred_score_name='score',
                           verbose=True):
        """
        This function computes several kpis of the model and return them in the form of a python dict.

        :param predictions: the predicted labels of the data
        :type predictions: pd.DataFrame
        :param true_class_name: name of column containing the true class labels, defaults to 'true_class'
        :type true_class_name: str, optional
        :param pred_class_name: name of column containing the predicted class labels, defaults to 'predicted_class'
        :type pred_class_name: str, optional
        :param pred_score_name: name of column containing the predicted score labels, defaults to 'score'
        :type pred_score_name: str, optional
        :param verbose: if the computed kpis have to be printed on stdout, defaults to True
        :type verbose: bool, optional
        """

        if any([not _c in predictions.columns.values for _c in [true_class_name, pred_class_name, pred_score_name]]):
            raise ValueError('The provided predictions data frame has not all the required columns.')

        kpis = {}

        kpis['confusion_matrix'] = confusion_matrix(predictions[true_class_name], predictions[pred_class_name])
        kpis['ROC_AUC'] = roc_auc_score(y_true=predictions[true_class_name], y_score=predictions[pred_score_name])
        kpis['precision'] = precision_score(y_true=predictions[true_class_name], y_pred=predictions[pred_class_name])
        kpis['average_precision_score'] = average_precision_score(y_true=predictions[true_class_name],
                                                                      y_score=predictions[pred_score_name])

        kpis['recall'] = recall_score(y_true=predictions[true_class_name], y_pred=predictions[pred_class_name])


        kpis['percentage_predicted_positives'] = predictions[pred_class_name].sum() / len(predictions)
        kpis['percentage_true_positives'] = predictions[true_class_name].sum() / len(predictions)

        if verbose:
            print('* * * Confusion matrix (true vs. pred) is: \n {0}'.format(kpis['confusion_matrix']))
            print('* * * AUC score is {0}'.format(kpis['ROC_AUC']))
            print('* * * AP is {0}'.format(kpis['average_precision_score']))
            print('* * * Precision score is {0}'.format(kpis['precision']))
            print('* * * Recall score is {0}'.format(kpis['recall']))
            print('* * * Predicted percentage of positives {0}'.format(kpis['percentage_predicted_positives']))
            print('* * * True percentage of positives {0}'.format(kpis['percentage_true_positives']))

        return kpis