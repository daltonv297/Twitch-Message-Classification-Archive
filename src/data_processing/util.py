import csv

import numpy as np
import pandas as pd
import json
import datetime
import string

BOT_LIST = ['moobot', 'nightbot', 'ohbot',
                        'deepbot', 'ankhbot', 'vivbot',
                        'wizebot', 'coebot', 'phantombot',
                        'xanbot', 'hnlbot', 'streamlabs',
                        'stay_hydrated_bot', 'botismo', 'streamelements',
                        'slanderbot', 'fossabot']

def load_streamer_chat(streamer):
    
    """ Returns a dataframe for a given streamer's chat

    Args:
        streamer: a string containing the name of the streamer

    Returns:
        a dataframe of messages from a given streamer. It's formatted as follows:
        text: Normalized chat message
        channel_name: Name of the channel
        timestamp: datetime object ("2022-11-16 16:03:17,942")
        username: name of the user who sent the message
        is_@: True if message starts with @ not targeted at streamer, False otherwise
        is_all_caps: True if message is in all caps, False otherwise
        punct_count: numpy array which stores counts for [!, ?, .]
    """
    # Returns true if starts with @ and doesn't @ the streamer
    def check_if_at(row):
        text = row["text"]
        return text[0] == '@' and text[1:1+len(streamer)] != streamer

    # Returns an list of punctuation counts based on string.punctuation
    # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    def punct_count(row):
        text = row["text"]
        punct_count_list = [None] * len(string.punctuation)
        for i in range(len(string.punctuation)):
            punct_count_list[i] = text.count(string.punctuation[i])
        return punct_count_list

    csv_path = '../../twitch-listener/logs/'+ streamer +'.csv'
    df = pd.read_csv(csv_path) # Reads in text, username, timestamp (all strings)

    # Clean data
    # TODO: Filter links
    df = df.dropna()
    df = df[~df["username"].isin(BOT_LIST)]
    df["is_all_caps"] = df.apply(lambda row: row.text.isupper(), axis=1)
    df["is_@"] = df.apply(lambda row: check_if_at(row), axis=1)
    df["punct_count"] = df.apply(lambda row: punct_count(row), axis=1)
    df["text"] = df["text"].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower()) # Set messages to lowercase and removes punct
    df["timestamp"] = df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S,%f')) # Converts timestamp string to timestamp datetime objects
    df["channel_name"] = len(df) * [streamer]

    return df
    
def create_input_example_pairs(streamer_df, list_of_other_dfs):
    """ Returns a list of close input example pairs for a given streamer
    
    """
    def similarity_score(timestamp1, timestamp2):
        difference = timestamp1 - timestamp2
        if difference.seconds < 3:
            return 0.95
        if difference.seconds < 5:
            return 0.8
        if difference.seconds < 20:
            return 0.6
        return 0.5

    WINDOW_SIZE = 5
    NUM_ADVERSARIAL = 5
    ADVERSARIAL_SCORE = 0.1
    SEED = 229
    np.random.seed(SEED)
    list_of_pairs = []

    streamer_df["timestamp"] = streamer_df["timestamp"].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')) # Converts timestamp string to timestamp datetime objects
    
    for row in streamer_df.index:
        s1 = streamer_df['text'][row]
        # Create close input pairs
        ts1 = streamer_df['timestamp'][row]
        if row < len(streamer_df)-WINDOW_SIZE:
            for i in range(1, WINDOW_SIZE):
                s2 = streamer_df['text'][row + i]
                ts2 = streamer_df['timestamp'][row + i]
                list_of_pairs.append((s1, s2, similarity_score(ts1, ts2)))

        # Create far input pairs
        other_random_streamers = np.random.randint(len(list_of_other_dfs), size = NUM_ADVERSARIAL)
        for j in range(NUM_ADVERSARIAL):
            other_streamer_df = list_of_other_dfs[other_random_streamers[j]]
            random_message = other_streamer_df['text'][np.random.randint(len(other_streamer_df))]
            list_of_pairs.append( (s1, random_message, ADVERSARIAL_SCORE) )
    
    return list_of_pairs 

def load_messages(streamer_list, test_ratio):
    """Returns a list of dataframes from streamers in streamer_list

    Args:
        streamer_list: a list of strings representing streamer names
        train_ratio: integer specifiying the percentage of data to assign to test data

    Returns:
        list_of_dataframes: a list of dataframes of messages from streamer_list
    """

    list_of_dataframes = []

    for streamer in streamer_list:
        dataframe = load_streamer_chat(streamer)
        #length = len(dataframe)
        list_of_dataframes.append(dataframe)

        #train_messages_len = length - length//test_ratio  # Reserve the last test_ratio percent of each streamer chat for test

        #train_dataframes = list_of_dataframes[:train_messages_len] 
        #test_dataframes = list_of_dataframes[train_messages_len: ]

    return list_of_dataframes #, train_dataframes, test_dataframes