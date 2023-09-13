import pandas as pd
import numpy as np
import random
import pickle

def create_ranking_sets(rankings):
  # Creates sets of 10 columns for every ranking question.
    lst = list(range(121))
    lst = lst[0::10]

    ranking_sets = []
    for i in lst:
        # select columns
        rank_set = rankings.iloc[:, i:i+10]
        # drop empty rows
        rank_set = rank_set.dropna()
        
        # add to list
        ranking_sets.append(rank_set)

    return ranking_sets


def transform_data(data, outliers,idx,n=18):
    column_mapping = {}
    for i, old_name in enumerate(data.columns):
        new_name = str(data.iloc[0,i])[-n:]  
        new_name = new_name 
        column_mapping[old_name] = new_name
    # Rename columns using the dictionary
    df = data.rename(columns=column_mapping)
    # Drop unnecessary rows
    df = df.drop(labels=0, axis=0)
    df = df.drop(labels=1, axis=0)
    outlier_lst = outliers.get(idx)
    # print(outlier_lst)
    if outlier_lst is not None:
        df = df.drop(df.index[[outlier_lst]], axis=0)

    lst_rows = []
    for i in range(df.shape[0]):
        row = df.iloc[i,:]
        lst_rows.append(row)

    data_ranked = pd.concat(lst_rows)

    return data_ranked

def create_ranking_csv(rankings, outliers):
    ranking_sets = create_ranking_sets(rankings)
 
    final_set =[]
    for i, set in enumerate(ranking_sets):
        # insert outlier removal function here
        
        # Transform the data
        data_ranked_new = transform_data(set, outliers, idx=i)
        print("{}-set shape: {}".format(i, data_ranked_new.shape))
        final_set.append(data_ranked_new)

    data_ranked = pd.concat(final_set)
  
    print("Final dataframe shape: ", data_ranked.shape)
    #data_ranked.to_csv('ranking_dataset_outlier_rem.csv', index=True, header=None)

    return data_ranked

def create_train_test():
    data = pd.read_csv('data/ranking_dataset_outlier_rem.csv', sep=',', header=None)
    set0 = np.arange(0,70,1)
    set1 = np.arange(730,800,1)
    set2 = np.arange(890,960,1)
    set3 = np.arange(1050,1120,1)

    test_indices = np.concatenate((set0,set1,set2,set3))
    test_data = data.iloc[test_indices]
    train_indices = [i for i in range(len(data.index)) if i not in test_indices]
    train_data = data.iloc[train_indices]
    
    print("Final train dataframe shape: ", train_data.shape)
    print("Final test dataframe shape: ", test_data.shape)
    train_data.to_csv('data/train.csv', index=False, header=None)
    test_data.to_csv('data/test.csv', index=False, header=None)



def main():
    # data = pd.read_csv('data/data_csv.csv', sep=',')
    # print('Shape data before preprocessing: ',data.shape)
    # meta = data.iloc[:,:14]
    # rankings = data.iloc[:,14:-1]
    # print('Meta shape', meta.shape)
    # print('Rankings shape', rankings.shape)
    # print("Nan values: ",rankings.isna().sum().sum())
    # print("Total entries with nans: ", rankings.shape[0] * rankings.shape[1])

    # with open('data/outliers.pkl', 'rb') as fp:
    #     outliers = pickle.load(fp)
        
    

    # final_df = create_ranking_csv(rankings=rankings, outliers=outliers)

    # print(final_df.head())

    create_train_test()

    # df = pd.read_csv('ranking_dataset.csv', sep=',', header=None)
    # df.columns =['name', 'rank']
    # df['rank']= df['rank'].subtract(1)
    # df.to_csv('ranking_dataset_c.csv', index=False)
    # return final_df

if __name__ == '__main__':
    main()
    