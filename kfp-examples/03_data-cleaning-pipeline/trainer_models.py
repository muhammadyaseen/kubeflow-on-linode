def prepare_univariate_baseline_data():
    selected_features = [
        'search_views_per_day_in_stock',
    ]

    base_X = as24_cleaned_anomalies_removed[selected_features]
    base_Y = as24_cleaned_anomalies_removed[['product_tier']]

    X_train, Y_train, X_test, Y_test = get_training_and_test_data_for_classification_task(
    base_X,
    base_Y,
    selected_features=selected_features,
    train_pct=0.90,
    test_pct=0.10,
    resample = False,
    resample_factors=None,
    random_state=RANDOM_SEED
    )

    print("[Original] Train and test inputs shape:")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

def prepare_full_model_data():
    
    selected_features = [
        'search_views_per_day_in_stock',
        'detail_views_per_day_in_stock',
        'stock_days_cleaned',
        'age',
        'price_relative',
        # 'ctr_cleaned',
        # 'weekends_while_in_stock'
    ]

    base_X = as24_cleaned_anomalies_removed[selected_features]
    base_Y = as24_cleaned_anomalies_removed[['product_tier']]

    X_train, Y_train, X_test, Y_test = get_training_and_test_data_for_classification_task(
    base_X,
    base_Y,
    selected_features=selected_features,
    train_pct=0.90,
    test_pct=0.10,
    resample = False,
    resample_factors=None,
    random_state=RANDOM_SEED
    )

    print("Train and test inputs shape:")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    X_train_res, Y_train_res, X_test_res, Y_test_res = get_training_and_test_data_for_classification_task(
    base_X,
    base_Y,
    selected_features=selected_features,
    train_pct=0.90,
    test_pct=0.10,
    resample = True,
    resample_factors={
        "Plus":  3,
        "Premium": 3
    },
    random_state=RANDOM_SEED
    )

    print("[Original] Train and test inputs shape:")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    print()
    print("[Resampled] Train and test inputs shape:")
    print(X_train_res.shape, Y_train_res.shape, X_test_res.shape, Y_test_res.shape)

def univariate_baseline():

    hparams = [{"class_weight": ["balanced", None]}]

    baseline_1d_logistic_clf_with_grid_search = ClassificationTask(
        "baseline_1d_logistic_clf_with_grid_search",
        LogisticRegression,
        {'random_state':RANDOM_SEED},
        hparams
    )

    baseline_1d_logistic_clf_with_grid_search.train(X_train, Y_train.values.ravel(), verbose=1)

    ranks = baseline_1d_logistic_clf_with_grid_search.show_best_model()

def train_logistic_regression_model():

    hparams = [
        {
        "penalty": ["l1", "l2", "elasticnet", None],
        "C": [1., 1e-1, 1e-2],
        "class_weight": ["balanced"],
        "multi_class": ["ovr", "multinomial"],
        "l1_ratio": [0.25, 0.50, 0.75, 0.95]
        }
    ]

    logistic_clf_with_grid_search = ClassificationTask(
        "logistic_clf_with_grid_search",
        LogisticRegression,
        {'random_state':RANDOM_SEED},
        hparams
    )

    logistic_clf_with_grid_search.train(X_train, Y_train.values.ravel(), verbose=1)
    ranks = logistic_clf_with_grid_search.show_best_model()

    logistic_clf_with_grid_search.train(X_train_res, Y_train_res.values.ravel(), verbose=1)
    ranks = logistic_clf_with_grid_search.show_best_model()

def train_decision_tree_model():
    
    hparams = [
        {
        "max_depth": [5,6,7,8],
        "min_samples_split": [10, 20, 50]
        }
    ]

    dtree_clf_with_grid_search = ClassificationTask(
        "Decision Tree",
        DecisionTreeClassifier,
        {'random_state':RANDOM_SEED},
        hparams
    )

    dtree_clf_with_grid_search.train(X_train, Y_train.values.ravel(), verbose=1)
    ranks = dtree_clf_with_grid_search.show_best_model()

    dtree_clf_with_grid_search.train(X_train_res, Y_train_res.values.ravel(), verbose=1)
    ranks = dtree_clf_with_grid_search.show_best_model()

def train_gbt_model():
    
    hparams = [
        {
        "max_depth": [2,3],
        "max_iter": [50, 100, 200],
        "class_weight": ["balanced"]
        }
    ]

    gbt_clf_with_grid_search = ClassificationTask(
        "Gradient Boosted Trees",
        HistGradientBoostingClassifier,
        {'random_state':RANDOM_SEED},
        hparams
    )

    gbt_clf_with_grid_search.train(X_train, Y_train.values.ravel(), verbose=1)
    ranks = gbt_clf_with_grid_search.show_best_model()

    gbt_clf_with_grid_search.train(X_train_res, Y_train_res.values.ravel(), verbose=1)
    ranks = gbt_clf_with_grid_search.show_best_model()


def find_best_model():
    
    # TODO: 
    pass

def test_on_best_model():
        
    Y_test_predicted = best_model.task_model_with_grid_search_trainer.best_estimator_.predict(X_test)
    Y_train_predicted = best_model.task_model_with_grid_search_trainer.best_estimator_.predict(X_train)

    best_model.plot_confusion_matrix(
        Y_train,
        Y_train_predicted,
        Y_test,
        Y_test_predicted
    )

    y_scores = best_model.task_model_with_grid_search_trainer.best_estimator_.predict_proba(X_train)

    best_model.plot_roc_and_pr_curves(Y_train, y_scores)
    