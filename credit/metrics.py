from sklearn import metrics


def simple_metrics(*, y_true, y_pred, y_score):
    """
    Accuracy (ACC) = (TP + TN) / (TP+TN+FP+FN)
    Precision: TP / (TP + FP)
    Sensitivity (SN/REC/TPR) : TP / (TP+FN)
    Specificity (SP) or true negative rate (TNR): TN/ (TN+FP)
    False Positive Rate (FPR): FP / (TN+FP) = 1 – specificity
    """
    CM = metrics.confusion_matrix(y_true, y_pred)
    TP = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TN = CM[1][1]
    acc = metrics.accuracy_score(y_true, y_pred)
    pre = TP / (TP+FP)
    pre2 = metrics.precision_score(y_true, y_pred)
    rec = TP / (TP+FN)
    rec2 = metrics.recall_score(y_true, y_pred)
    spe = TN / (TN+FP)
    fpr = 1 - spe
    roc = metrics.roc_auc_score(y_true, y_score)

    msg = """
    Accuracy    : {:.2f}%
    Precision   : {:.2f}% / {:.2f}%
    SN/REC/TPR  : {:.2f}% / {:.2f}%
    SP/TNR      : {:.2f}%
    FPR         : {:.2f}%
    ROC AUC     : {:.2f}%
    """.format(acc*100, pre*100, pre2*100, rec*100,
               rec2*100, spe*100, fpr*100, roc*100
               )
    print(msg)


