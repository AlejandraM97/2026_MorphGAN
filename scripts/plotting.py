import numpy as np
import matplotlib.pyplot as plt

def plot_classification_metrics(classification_results):
    """
    Plots extended classification metrics (specificity, recall, precision, AUC) 
    for each threshold type in a grouped bar chart.
    
    Args:
        classification_results (dict): Dictionary where keys are threshold types 
            (e.g., 'x', 'latent', 'sum') and values are dictionaries with keys 
            'specificity', 'recall', 'precision', and 'auc'.
    """
    # Get the threshold types (categories)
    categories = list(classification_results.keys())
    specificity = [classification_results[cat]['specificity'] for cat in categories]
    recall = [classification_results[cat]['recall'] for cat in categories]
    precision = [classification_results[cat]['precision'] for cat in categories]
    # If AUC is None, use 0 (or you can filter it out)
    auc = [classification_results[cat]['auc'] if classification_results[cat]['auc'] is not None else 0 
           for cat in categories]

    x = np.arange(len(categories))
    width = 0.2  # width for each bar

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5*width, specificity, width, label='Specificity')
    rects2 = ax.bar(x - 0.5*width, recall, width, label='Recall')
    rects3 = ax.bar(x + 0.5*width, precision, width, label='Precision')
    rects4 = ax.bar(x + 1.5*width, auc, width, label='AUC')

    ax.set_ylabel('Scores')
    ax.set_title('Extended Classification Metrics by Threshold Type')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Annotate bars with their values.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # vertical offset in points
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plots a confusion matrix using sklearn's ConfusionMatrixDisplay.
    
    Args:
        y_true (list or array): True binary labels.
        y_pred (list or array): Predicted binary labels.
        title (str): Title for the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()
