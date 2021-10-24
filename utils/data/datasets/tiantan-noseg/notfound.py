import pandas as pd
import json

from preprocess.convert import output_dir

if __name__ == '__main__':
    patient_info = pd.read_csv(output_dir / 'patient_info.csv')
    cohort = json.load(open('cohort.json'))
    patients = {info['patient'] for info in cohort}
    for patient in patient_info['patient']:
        if patient not in patients:
            print(patient)
