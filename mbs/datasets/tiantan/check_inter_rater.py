from pathlib import Path

import monai
from monai.metrics import DiceMetric

A_dir = Path('inter-rater-test/A')
B_dir = Path('inter-rater-test/B')
dice_metric = DiceMetric()

def get_cohort(patients_dir: Path) -> list[str]:
    return [patient_path.name for patient_path in patients_dir.iterdir()]

keys = ['AT', 'ST']

loader = monai.transforms.Compose([
    monai.transforms.LoadImageD(keys),
    monai.transforms.AddChannelD(keys),
    monai.transforms.ToTensorD(keys),
])

def process(patient: str):
    a, b = map(loader, (
        {
            key: data_dir / patient / f'{key}.nii'
            for key in keys
        }
        for data_dir in [A_dir, B_dir]
    ))
    for key in keys:
        print(patient, key, a[key].max().item(), b[key].max().item())

def main():
    cohort = list(set(get_cohort(A_dir)) & set(get_cohort(B_dir)))
    print(len(cohort))
    for patient in cohort:
        process(patient)

if __name__ == '__main__':
    main()
