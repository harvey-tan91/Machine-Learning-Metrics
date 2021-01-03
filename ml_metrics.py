

def grid_search_result_summary(grid_search_obj):

    import pandas as pd
    pd.set_option('display.max_colwidth', -1)

    result = grid_search_obj.cv_results_
    result_df = pd.DataFrame({
        'Params' : result['params'],
        'Mean Train Score %': result['mean_train_score']*100,
        'Std Train Score %': result['std_train_score']*100,
        'Mean Test Score %': result['mean_test_score']*100,
        'Std Test Score %': result['std_test_score']*100
        })
    result_df.sort_values(by='Mean Test Score %', ascending=False, inplace=True)

    print('Grid Search Result Summary:')
    print(f"Best Estimator's params: {grid_search_obj.best_params_}")
    print(f"Best Estimator's score %: {grid_search_obj.best_score_ * 100}")
    print(f"Best Estimator's index: {grid_search_obj.best_index_}")
    print(f"Number of cross-validation: {grid_search_obj.cv}")
    print(f"Number of Parameters: {len(result['params'])}")
    print()
    print()
    print(f'Top 15 Parameters by Mean Test Score')
    print(result_df.head(15))
    

def classifier_performance_report(classifier, X, y):
    """
    Objective: To generate a report that display the key performance metrics of a binary classifier
    Parameters
    ----------
    classifier: Classifier object
    X: Input feature
    y: Target feature
    """    

    import sklearn.metrics as m
    import sklearn.model_selection as ms
    import matplotlib.pyplot as plt

    prediction = classifier.predict(X)

    try:
        prediction_score = ms.cross_val_predict(classifier, X, y, cv=3, method='decision_function')
    except:
        prediction_score = ms.cross_val_predict(classifier, X, y, cv=3, method='predict_proba')
        prediction_score = prediction_score[:, 1]

    precision, recall, threhold = m.precision_recall_curve(y, prediction_score)

    # metrics
    classifier_accuracy = m.accuracy_score(y, prediction)
    classifier_precision = m.precision_score(y, prediction)
    classifier_recall = m.recall_score(y, prediction)
    classifier_f1_score = m.f1_score(y, prediction)

    # report
    print("-----Performance Summary-----")
    print(f'Accuracy Rate: {round(classifier_accuracy * 100,1)}%')
    print(f'Precision Rate: {round(classifier_precision * 100,1)}%')
    print(f'Recall Rate: {round(classifier_recall * 100,1)}%')
    print(f'F1 Score: {round(classifier_f1_score * 100,1)}%')
    print()
    print('Misc. Info:')
    print(f'Number of data points in X: {len(X)}')
    print()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,7.5))
    fig.suptitle("Performance Summary", y=1.05, fontsize=16)

    m.plot_confusion_matrix(classifier, X, y, ax=ax1)
    ax1.set_title('Confusion Matrix')
    
    m.plot_precision_recall_curve(classifier, X, y, ax= ax2)
    ax2.set_title('Precision-Recall Curve')
    
    m.plot_roc_curve(classifier, X, y, ax=ax3)
    ax3.set_title('ROC Curve')
    ax3.plot([0,1],[0,1], linestyle='--', color='k', label='Random Classifier')
    ax3.legend()

    ax4.plot(threhold, precision[:-1], color='g', label='Precision')
    ax4.plot(threhold, recall[:-1], color ='r', label='Recall')
    ax4.set_title('Precision-Recall Threshold')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Precision / Recall Rate')
    ax4.legend()

    plt.tight_layout()
    plt.show()


def classifier_cv_report(classifier, X, y, n_kFold, no_of_repeats, random_state=False):
    """
    Objective: To generate a report that display the cross-validation results of a classifier
    Cross-validation results are based on repeated k-fold cross-validation technique.

    Parameters
    ----------
    classifier: Classifier object
    X: Input feature
    y: Target feature
    n_kFold: Number of k-Folds
    no_of_repeats: Number of times to repeat the k-Folds cross-validation process
    """
    import sklearn.model_selection as ms
    import numpy as np

    cv = ms.RepeatedKFold(n_splits= n_kFold, n_repeats= no_of_repeats, random_state=random_state)
    cv_information = ms.cross_validate(classifier, X, y, scoring=['accuracy', 'precision', 'recall', 'f1'], \
        cv= cv, return_train_score=True)

    train_accuracy = cv_information['train_accuracy']
    train_precision = cv_information['train_precision']
    train_recall = cv_information['train_recall']
    train_f1 = cv_information['train_f1']

    test_accuracy = cv_information['test_accuracy']
    test_precision = cv_information['test_precision']
    test_recall = cv_information['train_recall']
    test_f1 = cv_information['test_f1']

    print('Summary on Training Data:')
    print(f'Average Accuracy & SD: {round(np.mean(train_accuracy) * 100, 1)}% , {round(np.std(train_accuracy) * 100, 3)}%')
    print(f'Average Precision & SD: {round(np.mean(train_precision) * 100, 1)}% , {round(np.std(train_precision) * 100, 3)}%')
    print(f'Average Recall & SD: {round(np.mean(train_recall) * 100, 1)}% , {round(np.std(train_recall) * 100, 3)}%')
    print(f'Average F1 & SD: {round(np.mean(train_f1) * 100, 1)}% , {round(np.std(train_f1) * 100, 3)}%')
    print()
    print('=====================')
    print()
    print('Summary on Testing Data:')
    print(f'Average Accuracy & SD: {round(np.mean(test_accuracy) * 100, 1)}% , {round(np.std(test_accuracy) * 100, 3)}%')
    print(f'Average Precision & SD: {round(np.mean(test_precision) * 100, 1)}% , {round(np.std(test_precision) * 100, 3)}%')
    print(f'Average Recall & SD: {round(np.mean(test_recall) * 100, 1)}% , {round(np.std(test_recall) * 100, 3)}%')
    print(f'Average F1 & SD: {round(np.mean(test_f1) * 100, 1)}% , {round(np.std(test_f1) * 100, 3)}%')


def classifier_plot_learning_curve(classifier, X, y, n_kFold=5, no_of_repeats=2, random_state=False):
    """
    """
    
    import sklearn.model_selection as ms
    import numpy as np
    import matplotlib.pyplot as plt

    scoring_option = ['accuracy', 'precision', 'recall', 'f1']
    cv = ms.RepeatedKFold(n_splits=n_kFold, n_repeats=no_of_repeats, random_state=random_state)
    train_size_option = np.arange(0.1,1.1,0.1)

    train_size_container = []
    mean_train_score_container = []
    mean_test_score_container = []
    std_train_score_container = []
    std_test_score_containter = []

    for option in scoring_option:
        train_size, train_score, test_score = ms.learning_curve(classifier, X, y, train_sizes=train_size_option, cv=cv, scoring=option)
        train_score = [i*100 for i in train_score]
        test_score = [i*100 for i in test_score]

        train_size_container.append(train_size)
        mean_train_score_container.append([np.mean(i) for i in train_score])
        mean_test_score_container.append([np.mean(i) for i in test_score])

        std_train_score_container.append([np.std(i) for i in train_score])
        std_test_score_containter.append([np.std(i) for i in test_score])

    train_size, *_ = train_size_container

    acc_mean_train_score, preci_mean_train_score, recall_mean_train_score, f1_mean_train_score = np.array(mean_train_score_container)

    acc_mean_test_score, preci_mean_test_score, recall_mean_test_score, f1_mean_test_score = np.array(mean_test_score_container)

    std_acc_mean_train_score, std_preci_mean_train_score, std_recall_mean_train_score, std_f1_mean_train_score = np.array(std_train_score_container)

    std_acc_mean_test_score, std_preci_mean_test_score, std_recall_mean_test_score, std_f1_mean_test_score = np.array(std_test_score_containter)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12,7.5))
    fig.suptitle("Learning Curve", y=1.05, fontsize=16)

    # ax1
    # TRAIN
    ax1.plot(train_size, acc_mean_train_score, color='g', marker='o', label='train')
    ax1.fill_between(train_size, acc_mean_train_score - std_acc_mean_train_score, acc_mean_train_score + std_acc_mean_train_score, alpha= 0.2, color='g')
    # TEST
    ax1.plot(train_size, acc_mean_test_score, color='r', marker='o', label='test')
    ax1.fill_between(train_size, acc_mean_test_score - std_acc_mean_test_score, acc_mean_test_score + std_acc_mean_test_score, alpha= 0.2, color='r')

    ax1.set_title('Accuracy')
    ax1.set_xlabel('Train Size')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # ax2
    # TRAIN
    ax2.plot(train_size, preci_mean_train_score, color='g', marker='o', label='train')
    ax2.fill_between(train_size, preci_mean_train_score - std_preci_mean_train_score, preci_mean_train_score + std_preci_mean_train_score, alpha= 0.2, color='g')
    # TEST
    ax2.plot(train_size, preci_mean_test_score, color='r', marker='o', label='test')
    ax2.fill_between(train_size, preci_mean_test_score - std_preci_mean_test_score, preci_mean_test_score + std_preci_mean_test_score, alpha= 0.2, color='r')

    ax2.set_title('Precision')
    ax2.set_xlabel('Train Size')
    ax2.set_ylabel('Precision')
    ax2.legend()

    # ax3
    # TRAIN
    ax3.plot(train_size, recall_mean_train_score, color='g', marker='o', label='train')
    ax3.fill_between(train_size, recall_mean_train_score - std_recall_mean_train_score, recall_mean_train_score + std_recall_mean_train_score, alpha= 0.2, color='g')
    # TEST
    ax3.plot(train_size, recall_mean_test_score, color='r', marker='o', label='test')
    ax3.fill_between(train_size, recall_mean_test_score - std_recall_mean_test_score, recall_mean_test_score + std_recall_mean_test_score, alpha= 0.2, color='r')

    ax3.set_title('Recall')
    ax3.set_xlabel('Train Size')
    ax3.set_ylabel('Recall')
    ax3.legend()

    # ax4
    # TRAIN
    ax4.plot(train_size, f1_mean_train_score, color='g', marker='o', label='train')
    ax4.fill_between(train_size, f1_mean_train_score - std_f1_mean_train_score, f1_mean_train_score + std_f1_mean_train_score, alpha= 0.2, color='g')
    # TEST
    ax4.plot(train_size, f1_mean_test_score, color='r', marker='o', label='test')
    ax4.fill_between(train_size, f1_mean_test_score - std_f1_mean_test_score, f1_mean_test_score + std_f1_mean_test_score, alpha= 0.2, color='r')

    ax4.set_title('F1')
    ax4.set_xlabel('Train Size')
    ax4.set_ylabel('F1')
    ax4.legend()

    plt.tight_layout()
    plt.show()



def decision_tree_classifier_ccp_analysis(tree_estimator, X_train, y_train, X_test, y_test):
    """
    THIS IS A WORK-IN-PROGRESS
    
    """

    import sklearn.tree as tree
    import matplotlib.pyplot as plt    
    import sklearn.model_selection as ms
    import numpy as np
    import pandas as pd

    path = tree_estimator.cost_complexity_pruning_path(X_train, y_train)
    ccp_alpha = path.ccp_alphas[:-1] # last value is omitted as the largest value will prune all the leaves leaving us with only the root node

    tree_stored = []

    for alpha in ccp_alpha:
        classifier = tree.DecisionTreeClassifier(random_state=0, ccp_alpha= alpha)
        classifier.fit(X_train, y_train)
        tree_stored += [classifier]

    train_score = [classifier.score(X_train, y_train) for classifier in tree_stored]
    test_score = [classifier.score(X_test, y_test) for classifier in test_score]

    alpha_cross_validation = []

    for alpha in ccp_alpha:
        classifier = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
        classifier.fit(X_train, y_train)
        accuracy_score = ms.cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
        alpha_cross_validation.append([alpha, np.mean(accuracy_score), np.std(accuracy_score)])

    alpha_table = pd.DataFrame(alpha_cross_validation, columns=['alpha', 'mean_accuracy', 'std_accuracy'])




    print(f'Number of estimators created: {len(ccp_alpha)}')
    print()

    plt.figure(figsize=(10,7.5))
    plt.title('Relationship between Pruning Factor (Alpha) and Score')
    plt.xlabel('Alpha')
    plt.ylabel('Score')
    plt.plot(ccp_alpha, train_score, marker='o', label='Train', drawstyle='steps-post')
    plt.plot(ccp_alpha, test_score, marker='o', label='Test', drawstyle='steps-post')
    plt.legend()


def gini_impurity_single_node(x:int, y:int):
    """
    Objective: To calculate the Gini impurity value for a single node
    Parameters
    ----------
    x: number of instances in the positve class
    y: number of instances in the negative class
    """

    total = x + y
    gini_value = 1 - (x/total)**2 - (y/total)**2
    return gini_value


def gini_impurity_two_node(x1:int, y1:int, x2:int, y2:int):
    """
    Objective: To calculate the Gini impurity value for a split node
    Parameters
    ----------
    x1: number of instances in the positive-positive class
    y1: number of instances in the positive-negative class
    x2: number of instances in the negative-positive class
    y2: number of instances in the negative-negative class
    """
    
    total_1= x1 + y1
    total_2= x2 + y2
    grand_total= total_1 + total_2

    gini_value_1= 1 - (x1 / total_1)**2 - (y1 / total_1)**2
    gini_value_2= 1 - (x2 / total_2)**2 - (y2 / total_2)**2

    weighted_gini_value= (total_1 / grand_total)*gini_value_1 + (total_2 / grand_total)*gini_value_2
    return weighted_gini_value
































