import os
from typing import *

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc


class ClassificationReporter:
    def __init__(self, report_dir, target_names: List[str]):
        self.report_dir = report_dir
        self.target_names = target_names

        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.patients = {}

    def append(self, patient: str, logit: torch.FloatTensor, label: int):
        self.y_true.append(label)
        pred = logit.argmax().item()
        self.y_pred.append(pred)
        output = torch.softmax(logit, dim=0)
        self.y_score.append(output.detach().cpu().numpy())
        self.patients.setdefault(patient, {
            'label': label,
            'cnt': np.zeros(4, dtype=np.int)
        })['cnt'][pred] += 1

    def report(self):
        from sklearn.metrics import confusion_matrix
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        report = classification_report(y_true, y_pred, target_names=self.target_names, output_dict=True)
        all_auc = self.get_auc()
        for i, target_name in enumerate(self.target_names):
            report[target_name]['auc'] = all_auc[i]

        report['weighted avg']['accuracy'] = report.pop('accuracy')
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.report_dir, 'report.tsv'), sep='\t')

        patient_true = []
        patient_pred = []
        for patient, info in self.patients.items():
            patient_true.append(info['label'])
            patient_pred.append(info['cnt'].argmax())
        report = classification_report(patient_true, patient_pred, target_names=self.target_names, output_dict=True)
        report['weighted avg']['accuracy'] = report.pop('accuracy')
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.report_dir, 'patient.tsv'), sep='\t')

    def get_auc(self):
        y_true = np.eye(len(self.target_names))[self.y_true]
        y_score = np.array(self.y_score)
        ret = []
        for i in range(len(self.target_names)):
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
            ret.append(auc(fpr, tpr))
        return ret

    def plot_roc(self):
        y_true = np.eye(len(self.target_names))[self.y_true]
        y_score = np.array(self.y_score)
        plt.figure()
        lw = 2
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        for i, name in enumerate(self.target_names):
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
            plt.plot(fpr, tpr, lw=lw, label=f'{name}, AUC = %0.2f' % auc(fpr, tpr))
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.05)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curves')
            plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.report_dir, 'roc.pdf'))
        plt.savefig(os.path.join(self.report_dir, 'roc.png'))
        plt.show()
