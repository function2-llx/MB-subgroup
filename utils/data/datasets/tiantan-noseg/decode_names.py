from pathlib import Path
import re

import pandas as pd
import pydicom

def main():
    pattern = re.compile(r'[^a-z]+')
    info = pd.read_csv('subgroup.csv')
    seg_files = pd.read_excel('../tiantan/results.xlsx', index_col=0)
    name_map = {
        pattern.sub('', idx): idx
        for idx in seg_files.index
    }
    vis = set()
    for idx, (patient_code, subgroup) in info.iterrows():
        name = ''
        for dcm in (Path('dcm') / patient_code).rglob('*.dcm'):
            try:
                ds = pydicom.dcmread(dcm)
                name = pattern.sub('', str(ds.PatientName).lower())
                break
            except RuntimeError:
                pass
        info.loc[idx, 'name'] = name
        name = name_map.get(name, None)
        if name is None:
            continue
        if name in vis:
            print(name)
        vis.add(name)
        info.loc[idx, 'name(raw)'] = name
        info.loc[idx, seg_files.columns] = seg_files.loc[name]

    for raw_name in seg_files.index:
        if raw_name not in info['name(raw)'].values:
            idx = len(info)
            info.loc[idx, 'name(raw)'] = raw_name
            info.loc[idx, seg_files.columns] = seg_files.loc[raw_name]

    info.to_excel('info.xlsx', index=False)

if __name__ == '__main__':
    main()
