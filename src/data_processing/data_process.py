import csv
import util
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit', 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']

def create_input_example_pairs(list_of_dfs):
    """ Creates a list of input example pairs given a list of dataframes from different chats
        It first creates a list of I-E pairs that are "close" (from a small window within a given streamer chat)
        Then, it creates a list of I-E pairs that are "far" (pulls pairs from randoms dfs and assumes they're dissimilar)
        It then concatenates the two lists and returns

        Args: 
            list_of_dfs: a list of dataframes that contain chat information from different streamers
    """
    save_path = 'input_example_pairs/'
    list_of_pairs = []
    # First, create input example pairs of messages that are close
    for i in range(len(STREAMERS)):
        streamer_df = list_of_dfs[i]
        list_of_other_dfs = list_of_dfs[:i] + list_of_dfs[(i + 1):]
        list_of_pairs_to_append = util.create_input_example_pairs(streamer_df, list_of_other_dfs)
        print('Finished creating pairs for '+STREAMERS[i])

        # Save file to csv
        with open(save_path+STREAMERS[i]+ '.csv','w', encoding = "utf-8") as out:
            csv_out=csv.writer(out)
            csv_out.writerow(['message1','message2','similarity_score'])
            for row in list_of_pairs_to_append:
                csv_out.writerow(row)

        list_of_pairs.append(list_of_pairs_to_append)
    
    return list_of_pairs

def create_list_of_dataframes():
    list_of_dataframes = util.load_messages(STREAMERS, 30)
    
    # Save path
    save_path = 'cleaned_data/'

    list_of_train_df = []
    list_of_test_df = []
    list_of_valid_df = []

    # Save each dataframe as a csv file
    # Additionally, split each dataframe into a 80/10/10 split for train test valid
    for i in range(len(list_of_dataframes)):
        streamer_name = STREAMERS[i]
        list_of_dataframes[i].to_csv(save_path + streamer_name + '.csv')
        
        train, test = train_test_split(list_of_dataframes[i], test_size = 0.2)
        test, val = train_test_split(test, test_size = 0.5)
        list_of_train_df.append(train)
        list_of_test_df.append(test)
        list_of_valid_df.append(val)

    # Save combined dataframe as a csv file
    combined_dataframe = pd.concat(list_of_dataframes)
    combined_dataframe.to_csv(save_path + 'combined_dataframe.csv')

    # Save combined test, train, valid to a csv file
    df_comb_train = pd.concat(list_of_train_df)
    df_comb_test = pd.concat(list_of_test_df)
    df_comb_valid = pd.concat(list_of_valid_df)

    df_comb_train.to_csv(save_path + "combined_train.csv")
    df_comb_test.to_csv(save_path + "combined_test.csv")
    df_comb_valid.to_csv(save_path + "combined_valid.csv")

    return list_of_dataframes

def main():

    list_of_dataframes = []
    if os.path.exists('cleaned_data/AdinRoss.csv'):
        print("Clean data already exists, loading in now.")
        for i in range(len(STREAMERS)):
            path2csv = 'cleaned_data/'+STREAMERS[i]+'.csv'
            list_of_dataframes.append(pd.read_csv(path2csv, keep_default_na=False))
    else:
        print("Clean data has not yet been generated. Generating now.")
        list_of_dataframes = create_list_of_dataframes()

    print("Generating input_example_pairs")
    create_input_example_pairs(list_of_dataframes)

if __name__ == "__main__":
    main()