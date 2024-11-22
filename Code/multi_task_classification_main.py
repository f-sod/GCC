import pandas as pd
from multi_task_classification_forest_ensemble import MultiTaskForestEnsemble
import random
from sklearn.metrics import matthews_corrcoef
import numpy as np
import copy


def main():
    profiles_cid = pd.read_pickle('../Data/Output/profiles_with_CID.pkl')
    profiles_cid.drop(
        columns=['Metadata_broad_sample', 'CPD_NAME', 'CPD_SMILES'], inplace=True)
    profiles_cid.set_index('CID', drop=True, inplace=True)
    responses = pd.read_csv(
        '../Data/Output/activity_to_receptors.csv', sep='\t')
    responses.set_index('receptor', drop=True, inplace=True)
    responses.columns = responses.columns.astype('int')
    present_cids = set(responses.columns)
    present_cids = present_cids.intersection(set(profiles_cid.index))
    responses = responses.loc[:, list(present_cids)]
    profiles_cid = profiles_cid.loc[list(present_cids), :]

    profiles_cid = profiles_cid[~profiles_cid.index.duplicated(
        keep='first')]
    mfe = MultiTaskForestEnsemble(n_forests=8, class_weight='balanced', n_tree_range=[
                                  100, 200, 300, 400, 500], max_depth_range=[10, 20, 30], min_samples_leaf_range=[5, 10, 15])
    train_indices = {}
    test_indices = {}
    for i in range(8):
        responses_i = responses.iloc[i, :].dropna()
        profiles_i = profiles_cid.loc[list(responses_i.index), :]
        response_indeces_list = list(responses_i.index)
        random.Random(42).shuffle(response_indeces_list)
        train_size = int(len(response_indeces_list)*0.8)
        train_indices[i] = response_indeces_list[:train_size]
        test_indices[i] = response_indeces_list[train_size:]

    mfe.cross_validate_and_fit_forests(
        X_train=copy.deepcopy(profiles_cid), y_train=responses, train_indices=train_indices)
    predictions = mfe.predict(X_test=profiles_cid, test_indices=test_indices)
    for i, pred in enumerate(predictions):
        act = responses.iloc[i, :]
        act = act[test_indices[i]]
        print(f'{responses.index[i]}: {matthews_corrcoef(act.values, pred)}')


if __name__ == '__main__':
    main()
