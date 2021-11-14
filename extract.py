from radiomics.featureextractor import RadiomicsFeatureExtractor
import pandas as pd

def main():
    results = pd.DataFrame()
    extractor = RadiomicsFeatureExtractor('radiomics-params/baseline.yml')
    print(extractor.enabledFeatures)
    features = extractor.execute(
        'utils/data/datasets/tiantan/stripped/333739tangrui/2-T2.nii.gz',
        'utils/data/datasets/tiantan/origin/333739tangrui/AT.nii.gz',
    )
    features = pd.Series(features)
    features.name = 'test'
    results = results.join(features, how='outer')
    results.to_csv('test.csv')

if __name__ == '__main__':
    main()
