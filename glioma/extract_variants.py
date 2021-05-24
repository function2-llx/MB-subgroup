import gzip
from pathlib import Path
import json

import pandas as pd

if __name__ == '__main__':
    variants = {
        case_uuid: []
        for case_uuid in pd.read_excel('cases.xlsx')['case_uuid']
    }
    for filepath in Path('masked_somatic_mutations').glob('**/*.maf.gz'):
        with gzip.open(filepath) as f:
            for i in range(5):
                f.readline()
            df = pd.read_csv(f, sep='\t').set_index('case_id')
        for case_uuid, data in df[['Hugo_Symbol', 'Variant_Classification', 'HGVSc', 'HGVSp']].iterrows():
            variants[case_uuid].append(data.to_dict())

    json.dump(variants, open('variants.json', 'w'), indent=4, ensure_ascii=False)
