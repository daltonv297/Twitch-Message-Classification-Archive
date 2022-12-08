import csv
import util
import numpy as np
import pandas as pd
import os
import sys
import datetime
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
    save_path = 'input_example_pairs_chrono/'
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

def save_dataframes_to_csv(list_of_all_dataframes):
    """
    """
    
    # Save path
    save_path = 'cleaned_data/'

    # Save each dataframe as a csv file
    # Additionally, split each dataframe into a 80/10/10 split for train test valid
    for i in range(len(list_of_all_dataframes)):
        streamer_name = STREAMERS[i]
        streamer_df = list_of_all_dataframes[i]
        # list_of_dataframes[i].to_csv(save_path + streamer_name + '.csv') # Uncomment if you want a complete csv for literally no reason
        
        #train, test = train_test_split(LIST_OF_ALL_DATAFRAMES[i], test_size = 0.2)
        #test, val = train_test_split(test, test_size = 0.5)
        num_rows = len(streamer_df)
        train_size = (int) (num_rows * 0.7)
        valid_size = (int) ((num_rows - train_size)*0.66)
        test_size = num_rows - train_size - valid_size

        train = streamer_df[:train_size]
        val = streamer_df[train_size:train_size+valid_size]
        test = streamer_df[train_size+valid_size:]
        
        train.to_csv(save_path+'train_chrono/'+streamer_name + '.csv')
        test.to_csv(save_path+'test_chrono/'+streamer_name + '.csv')
        val.to_csv(save_path+'valid_chrono/'+streamer_name + '.csv')

def messages_in_window(window):
    """ Prints a list that represents the average number of messages per streamer
        within a given context window (given in seconds)
    """
    print('Calculating average messages for window of', window, 'seconds')
    
    buffer_size = 20
    path2csvs = ['cleaned_data/train_chrono/','cleaned_data/test_chrono/','cleaned_data/valid_chrono/']
    for i in range(len(STREAMERS)):
        streamer_name = STREAMERS[i]
        streamer_df = []
        for path in path2csvs:
            streamer_df.append(pd.read_csv(path+streamer_name+'.csv', keep_default_na=False))
        streamer_df = pd.concat(streamer_df)

        messages = streamer_df['text'].tolist()
        timestamp_strings = streamer_df['timestamp'].tolist()
        timestamps = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in timestamp_strings] # List of timestamp objects

        tot_messages = 0
        num_of_messages_in_window = 0

        for j in range(buffer_size, len(messages)-buffer_size):
            tot_messages += 1
            cur_ts = timestamps[j]
            reached_prev_limit = False
            reached_post_limit = False
            prev_limit_index = j-1
            post_limit_index = j+1
            # Find the earliest message to include in context
            while not reached_prev_limit and prev_limit_index >= 0:
                prev_message_ts = timestamps[prev_limit_index]
                #print((cur_ts - prev_message_ts).seconds)
                if (cur_ts - prev_message_ts).seconds < window:
                    prev_limit_index -= 1
                else:
                    reached_prev_limit = True
            # Find latest message to include in context
            while not reached_post_limit and post_limit_index < len(messages):
                post_message_ts = timestamps[post_limit_index]
                #print((post_message_ts - cur_ts).seconds)
                if (post_message_ts - cur_ts).seconds < window:
                    post_limit_index += 1
                else:
                    reached_post_limit = True
            num_of_messages_in_window += (j-prev_limit_index) + (post_limit_index-j)
        
        print('Average number of messages for', streamer_name, 'in a duration window of', window,'seconds is', (num_of_messages_in_window/tot_messages))

def main():
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == '-window':
        window_size = int(args[1])
        messages_in_window(window_size)
    
    else:
        list_of_train_dataframes = []
        data_generated = os.path.exists('cleaned_data/train_chrono/AdinRoss.csv')
        if not data_generated:
            print("Clean data has not yet been generated. Generating now.")
            list_of_all_dataframes =  util.load_messages(STREAMERS)
            save_dataframes_to_csv(list_of_all_dataframes)
            print("Finished creating clean data. Loading in training data now.")
        else:
            print("Clean data already exists, loading in training data now.")

        for i in range(len(STREAMERS)):
            path2csv = 'cleaned_data/train_chrono/'+STREAMERS[i]+'.csv'
            list_of_train_dataframes.append(pd.read_csv(path2csv, keep_default_na=False))

        print("Generating input_example_pairs on train_dataframes")
        create_input_example_pairs(list_of_train_dataframes)

if __name__ == "__main__":
    main()