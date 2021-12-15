from __future__ import annotations

import os
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix

class Reporter:
    def __init__(self, report_dir, target_names: List[str], seg_names: List[str] = None):
        self.report_dir = Path(report_dir)
        self.target_names = target_names
        self.seg_names = seg_names

        self.y_true = []
        self.y_pred = []
        self.y_score = []
        self.results = []
        self.meandices = []

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
        fig, ax = plt.subplots()
        lw = 2
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        for i, name in enumerate(self.target_names):
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
            ax.plot(fpr, tpr, lw=lw, label=f'{name}, AUC = %0.2f' % auc(fpr, tpr))
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC curves')
            ax.legend(loc="lower right")
        fig.savefig(self.report_dir / 'roc.pdf')
        fig.savefig(self.report_dir / 'roc.png')
        plt.show()
        plt.close()

    def report(self):
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.plot_roc()
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        meandices = torch.stack(self.meandices)
        report = self.get_report(y_pred, y_true)
        pd.DataFrame(report).transpose().to_csv(self.report_dir / 'report.csv')
        pd.DataFrame.from_records(
            self.results,
            columns=['patient', 'ref', 'pred', *self.target_names],
        ).to_csv(self.report_dir / 'cls.csv', index=False)

        pd.DataFrame(confusion_matrix(y_true, y_pred), index=self.target_names, columns=self.target_names).to_csv(self.report_dir / 'cm.csv')
        pd.DataFrame({
            **{
                str(patient): {
                    seg_name: meandice[i].item()
                    for i, seg_name in enumerate(self.seg_names)
                }
                for (patient, *_), meandice in zip(self.results, meandices)
            },
            'average': {
                seg_name: meandices[:, i].mean().item()
                for i, seg_name in enumerate(self.seg_names)
            },
            'min': {
                seg_name: meandices[:, i].min().item()
                for i, seg_name in enumerate(self.seg_names)
            },
            'max': {
                seg_name: meandices[:, i].max().item()
                for i, seg_name in enumerate(self.seg_names)
            },
        }).transpose().to_csv(self.report_dir / 'meandice.csv')

    def digest(self, metric_names: List[str] = None):
        if metric_names is None:
            metric_names = ['precision', 'recall', 'auc']
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        meandices = torch.stack(self.meandices)
        report = self.get_report(y_pred, y_true)
        return {
            'acc': report['weighted avg']['accuracy'],
            **{
                f'{label}-{metric}': report[label][metric]
                for label in self.target_names for metric in metric_names
            },
            **{
                f'{seg_name} mDice': meandices[:, i].mean().item()
                for i, seg_name in enumerate(self.seg_names)
            },
        }

    def get_report(self, y_pred, y_true) -> Dict[str, Dict]:
        report = classification_report(y_true, y_pred, target_names=self.target_names, output_dict=True, zero_division=0)
        all_auc = self.get_auc()
        for i, target_name in enumerate(self.target_names):
            report[target_name]['auc'] = all_auc[i]
        report['weighted avg']['accuracy'] = report.pop('accuracy')
        return report

    def append(self, data, prob_pred: torch.FloatTensor, meandice: torch.FloatTensor):
        patient = data['patient'][0]
        label = data['label'].item()
        self.y_true.append(label)
        pred = prob_pred.argmax().item()
        self.y_pred.append(pred)
        score = prob_pred.detach().cpu().numpy()
        self.y_score.append(score)
        self.results.append((patient, self.target_names[label], self.target_names[pred], *score))
        self.meandices.append(meandice)

    def append_pred(self, pred: int, label: int):
        self.y_true.append(label)
        self.y_pred.append(pred)
        self.y_score.append(np.eye(len(self.target_names))[pred])

class Reported2d(Reporter):
    def __init__(self, report_dir, target_names: List[str]):
        super().__init__(self, report_dir, target_names)
        self.patients = {}

    def append_slice(self, patient: str, logit: torch.FloatTensor, label: int):
        self.y_true.append(label)
        pred = logit.argmax().item()
        self.y_pred.append(pred)
        output = torch.softmax(logit, dim=0)
        self.y_score.append(output.detach().cpu().numpy())
        self.patients.setdefault(patient, {
            'label': label,
            'cnt': np.zeros(4, dtype=int)
        })['cnt'][pred] += 1

    def report(self):
        super().report()
        patient_true = []
        patient_pred = []
        for patient, info in self.patients.items():
            patient_true.append(info['label'])
            patient_pred.append(info['cnt'].argmax())

        report = classification_report(patient_true, patient_pred, target_names=self.target_names, output_dict=True)
        report['weighted avg']['accuracy'] = report.pop('accuracy')
        pd.DataFrame(report).transpose().to_csv(os.path.join(self.report_dir, 'patient.tsv'), sep='\t')
