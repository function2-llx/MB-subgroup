import itertools
from pathlib import Path

import pandas as pd
import torch
from lightning.fabric import Fabric

from luolib.utils import import_object
from mbs.datamodule import load_split, MBClsDataModule
from mbs.models import MBClsModel
from mbs.utils.enums import MBDataKey, SUBGROUPS

class MBTester:
    """A class for ensemble prediction using multiple trained models.
    
    Args:
        model_cls_path: Import path to the model class
        num_folds: Number of cross-validation folds
        p_seeds: List of random seeds used in training
        output_dir: Directory containing the model checkpoints
    """
    def __init__(
        self,
        *,
        model_cls_path: str,
        num_folds: int = 5,
        p_seeds: list[int],
        output_dir: Path,
    ):
        model_cls = import_object(model_cls_path)
        assert issubclass(model_cls, MBClsModel)
        
        # Initialize models for each fold and seed
        self.models: list[list[MBClsModel]] = [[] for _ in range(num_folds)]
        for seed, fold_id in itertools.product(p_seeds, range(num_folds)):
            ckpt_path = output_dir / f'run-{seed}' / f'fold-{fold_id}' / 'last.ckpt'
            self.models[fold_id].append(model_cls.load_from_checkpoint(ckpt_path, strict=True))
            print(f'load model from {ckpt_path}')

        self.split = load_split()

    def predict_batch(self, batch: dict, fabric: Fabric) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict on a batch of data using ensemble of models.
        
        Args:
            batch: Input batch containing case IDs and model inputs
            fabric: Fabric instance to run inference on
            
        Returns:
            Tuple of (predictions, probabilities) tensors
        """
        cases = batch[MBDataKey.CASE]
        device = fabric.device
        
        # Check if all cases are from the same split
        sample_case = cases[0]
        split = self.split[sample_case]
        for case in cases[1:]:
            assert split == self.split[case]

        prob = None
        pred = torch.empty((len(cases), len(self.models)), dtype=torch.int32, device=device)
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                logit = model.infer_logit({k: v.to(device) if torch.is_tensor(v) else v 
                                         for k, v in batch.items()})
                cur_prob = logit.softmax(dim=-1)
                pred[:, i] = cur_prob.argmax(dim=-1)
                if prob is None:
                    prob = cur_prob
                else:
                    prob += cur_prob
                    
        prob /= len(self.models)
        return pred, prob


def main(
    datamodule: MBClsDataModule,
    model_cls: str,
    *,
    num_folds: int = 5,
    p_seeds: list[int] = [42],
    output_dir: Path,
    precision: str = "32-true",
):
    """Run inference using trained models.
    
    Args:
        datamodule: Data module containing the test dataset
        model_cls: Import path to model class
        num_folds: Number of cross-validation folds
        p_seeds: List of random seeds used in training
        output_dir: Directory containing model checkpoints
        precision: Precision to use for computation
    """
    # Initialize Fabric
    fabric = Fabric(precision=precision)
    fabric.launch()
    
    # Initialize tester and dataset
    tester = MBTester(
        model_cls_path=model_cls,
        num_folds=num_folds,
        p_seeds=p_seeds,
        output_dir=output_dir
    )
    
    test_loader = datamodule.predict_dataloader()
    fabric.setup
    
    # Run predictions
    all_probs = []
    all_cases = []
    all_preds = []
    for batch in fabric.iterator(test_loader):
        pred, prob = tester.predict_batch(batch, fabric)
        all_probs.append(prob.cpu())
        all_preds.append(pred.cpu())
        all_cases.extend(batch[MBDataKey.CASE])

    # Concatenate all probabilities
    all_probs = torch.cat(all_probs, dim=0)
    
    # Create DataFrame with predictions
    results = {}
    results['case'] = all_cases
    results['pred'] = all_preds
    for i, subgroup in enumerate(SUBGROUPS):
        results[subgroup] = all_probs[:, i].numpy()

    # Save to Excel file
    df = pd.DataFrame(results)
    output_path = output_dir / 'predictions.xlsx'
    df.to_excel(output_path, index=False)

if __name__ == '__main__':
    main()
