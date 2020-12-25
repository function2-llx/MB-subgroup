from typing import *

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


class ClassificationReporter:
    def __init__(self, classes: List):
        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.classes = classes

    def append(self, logit: torch.FloatTensor, label: int):
        self.y_true.append(label)
        self.y_pred.append(logit.argmax().item())
        output = torch.softmax(logit, dim=0)
        self.y_score.append(output.detach().cpu().numpy())

    def report(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        return classification_report(y_true, y_pred, output_dict=True)

    def roc(self):
        fig, axs = plt.subplots(1, len(self.classes))
        for i, cls, ax in enumerate(zip(self.classes, axs)):
            cls = str(cls)
            fpr, tpr, thresholds = roc_curve(self.y_true[:, i], self.y_score[:, i])
            lw = 2
            ax.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
            ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax.xlim([0.0, 1.0])
            ax.ylim([0.0, 1.05])
            ax.xlabel('False Positive Rate')
            ax.ylabel('True Positive Rate')
            ax.title(cls)
            ax.legend(loc="lower right")
        return fig
