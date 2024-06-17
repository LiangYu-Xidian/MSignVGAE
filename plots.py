import matplotlib.pyplot as plt
from sklearn import metrics



def plot_ROC(y_true, y_pred, y_pred_proba, path='./fig/temp_ROC.tif'):
    auc_score = metrics.roc_auc_score(y_true, y_pred_proba)
    acc_score = metrics.accuracy_score(y_true, y_pred)
    pre_score = metrics.precision_score(y_true, y_pred)
    rec_score = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_proba)
    auc = metrics.auc(fpr, tpr)

    plt.figure(1)
    lw = 2

    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(path, dpi=300)
    plt.close()

    return auc_score, acc_score, pre_score, rec_score, f1_score


def plot_PR(y_true, y_pred, y_pred_proba, path='./fig/temp_PR.tif'):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred_proba)
    aupr = metrics.auc(recall, precision)

    plt.figure(2)
    lw = 2

    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % aupr)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(path, dpi=300)

    return aupr
