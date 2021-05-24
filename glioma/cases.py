import json

import pandas as pd

cases = {
    case['case_id']: {
        'case_uuid': case['case_id'],
        'case_id': case['submitter_id'],
        'project': case['project']['project_id'],
    }
    for case in json.load(open('cases.json'))
}
for data in json.load(open('clinical.json')):
    case = cases[data['case_id']]

    if 'demographic' in data:
        demographic = data['demographic']
        for k in ['gender', 'vital_status', 'race', 'days_to_birth', 'days_to_death']:
            if k in demographic:
                case[k] = demographic[k]

    if 'diagnoses' in data:
        diagnoses = data['diagnoses']
        assert len(diagnoses) == 1
        diagnoses = diagnoses[0]
        for k in ['tissue_or_organ_of_origin', 'primary_diagnosis', 'prior_malignancy', 'year_of_diagnosis', 'prior_treatment']:
            case[k] = diagnoses[k]

df = pd.DataFrame(cases.values())

filename = 'cases.xlsx'
writer = pd.ExcelWriter(filename, engine='xlsxwriter')
for sheetname, df in {'cases': df}.items():  # loop through `dict` of dataframes
    df.to_excel(writer, sheet_name=sheetname, index=False)  # send df to writer
    worksheet = writer.sheets[sheetname]  # pull worksheet object
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len(str(series.name))  # len of column name/header
            )) + 1  # adding a little extra space
        worksheet.set_column(idx, idx, max_len + 1)  # set column width
writer.save()
