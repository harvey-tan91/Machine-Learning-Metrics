

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
    Objective: To generate a report that display the key metrics on the classifier's performance
    Parameters:
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
    

def decision_tree_classifier_ccp_analysis(tree_estimator, X_train, y_train, X_test, y_test):
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
    Parameters:
    x: number of instances in the positve class
    y: number of instances in the negative class
    """

    total = x + y
    gini_value = 1 - (x/total)**2 - (y/total)**2
    return gini_value


def gini_impurity_two_node(x1:int, y1:int, x2:int, y2:int):
    """
    Objective: To calculate the Gini impurity value for a split node
    Parameters:
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

































