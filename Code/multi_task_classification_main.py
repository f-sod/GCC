import pandas as pd
from multi_task_classification_forest_ensemble import MultiTaskForestEnsemble
import random
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score
import numpy as np
import copy
from sklearn.model_selection import StratifiedKFold


def main():
    for feature_type in  ['morphological', 'structural']:
        if feature_type == 'morphological':
            profiles_cid = pd.read_pickle('../Data/Output/profiles_with_CID.pkl')

            profiles_cid.drop(
                columns=['Metadata_broad_sample', 'CPD_NAME', 'CPD_SMILES'], inplace=True)
        elif feature_type == 'structural':
            profiles_cid = pd.read_csv('../Data/CellProfiles/structural_fps_matrix.csv', sep = '\t')

        profiles_cid.set_index('CID', drop=True, inplace=True)
        for response_type in ['agonists', 'antagonists', 'all']:
            print(f'{feature_type} and {response_type}')
            if response_type == 'all':
                responses = pd.read_csv(
                    '../Data/Output/activity_to_receptors.csv', sep='\t')
            else:
            
                responses = pd.read_csv('../Data/Output/agonists.csv', sep='\t')
            responses.set_index('receptor', drop=True, inplace=True)

            if response_type == 'antagonists':
                responses.drop(index='rxr', inplace=True)
                n_forests = 7
            else:
                n_forests = 8

            responses.columns = responses.columns.astype('int')
            responses = responses.astype('float')
            present_cids = set(responses.columns)
            present_cids = present_cids.intersection(set(profiles_cid.index))
            if 679 in present_cids:
                # remove DMSO (CID = 679)
                present_cids.remove(679)
            responses = responses.loc[:, list(present_cids)]
            profiles_cid = profiles_cid.loc[list(present_cids), :]

            profiles_cid = profiles_cid[~profiles_cid.index.duplicated(
                keep='first')]
        
            mfe = MultiTaskForestEnsemble(n_forests=n_forests, class_weight='balanced', n_tree_range=[
                                            100, 200, 300, 400, 500], max_depth_range=[10, 20, 30], min_samples_leaf_range=[5, 10, 15])
            train_indices = {}
            test_indices = {}

            cv_folds = []
            for i in range(n_forests):
                responses_i = responses.iloc[i, :].dropna()
                profiles_i = profiles_cid.loc[list(responses_i.index), :]
                response_indeces_list = list(responses_i.index)
                random.Random(42).shuffle(response_indeces_list)
                train_size = int(len(response_indeces_list)*0.8)
                train_indices[i] = response_indeces_list[:train_size]
                test_indices[i] = response_indeces_list[train_size:]
                receptor_name = responses.index[i]
                profiles_train = profiles_i.loc[train_indices[i], :]
                profiles_train.to_csv(
                    f'../Data/Output/{receptor_name}_{feature_type}_{response_type}_X_train.csv', sep='\t')
                responses_train = responses_i[train_indices[i]]
                responses_train.to_csv(
                    f'../Data/Output/{receptor_name}_{feature_type}_{response_type}_y_train.csv', sep='\t')

                profiles_test = profiles_i.loc[test_indices[i], :]
                profiles_test.to_csv(
                    f'../Data/Output/{receptor_name}_{feature_type}_{response_type}_X_test.csv', sep='\t')
                responses_test = responses_i[test_indices[i]]
                responses_test.to_csv(
                    f'../Data/Output/{receptor_name}_{feature_type}_{response_type}_y_test.csv', sep='\t')

                skf = StratifiedKFold(
                    n_splits=5, shuffle=True, random_state=42)
                folds = skf.split(X=profiles_train, y=responses_train)
                cv_folds.append(folds)

            '''
            with open('../Data/Output/feature_selection_files_MRMR.txt', 'r') as feature_selection_files:
                files = feature_selection_files.read().splitlines()
                features_for_training = []
                for file in files:
                    fs_dataframe = pd.read_csv(file, sep='\t')
                    features = fs_dataframe.iloc[:, 1].values
                    top_50_features = features
                    features_for_training.append(top_50_features)
            '''


            features_for_training = None
            mfe.cross_validate_and_fit_forests(
                X_train=copy.deepcopy(profiles_cid), y_train=responses, train_indices=train_indices, train_feature_list=features_for_training,
                cv_folds=None)
            predictions_test = mfe.predict(
                X_test=profiles_cid, test_indices=test_indices, train_feature_list=features_for_training)
            predictions_train = mfe.predict(
                X_test=profiles_cid, test_indices=train_indices, train_feature_list=features_for_training)

            output_path = f'../Data/Output/results_receptor_specific_models_{response_type}_{feature_type}_fps.csv'
            with open(output_path, 'w') as output:
                output.write(f'set\treceptor\tMCC\tbalanced acc\tAUC\n')
            for i, pred in enumerate(predictions_test):
                act = responses.iloc[i, :]
                act_test = act[test_indices[i]]
                act_train = act[train_indices[i]]
                with open(output_path, 'a') as output:
                    test_mcc = matthews_corrcoef(act_test.values, pred)
                    train_mcc = matthews_corrcoef(
                        act_train.values, predictions_train[i])
                    test_balanced_acc = balanced_accuracy_score(act_test.values, pred)
                    train_balanced_acc = balanced_accuracy_score(
                        act_train.values, predictions_train[i])
                    test_auc = roc_auc_score(act_test.values, pred)
                    train_auc = roc_auc_score(act_train.values, predictions_train[i])
                    output.write(
                        f'test\t{responses.index[i]}\t{test_mcc}\t{test_balanced_acc}\t{test_auc}\n')
                    output.write(
                        f'train\t{responses.index[i]}\t{train_mcc}\t{train_balanced_acc}\t{train_auc}\n')
                


if __name__ == '__main__':
    main()
