import csv

import numpy as np
import pandas as pd
import json


def load_streamer_chat(streamer):
    
    """ Returns a list of messages from a given streamer's logs

    Args:
        streamer: a string containing the name of the streamer

    Returns:
        a list of messages from a given streamer
    """
    csv_path = 'twitch-listener/'+ streamer +'.csv'

    messages = []
    
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            messages.append(row[0])

    return messages[1:]  # Initial first message is just the 'text' header


def load_messages(streamer_list, test_ratio):
    """Returns a list of messages from streamers in streamer_list, as well as labels

    Args:
        streamer_list: a list of strings representing streamer names
        train_ratio: integer specifiying the percentage of data to assign to test data

    Returns:
        all_messages: a list of all messages from streamer_list
        all_labels: a list of corresponding labels
    """

    train_messages = [] # A list of all messages from all the chat
    train_labels = [] # A corresponding list of labels. Labelled with an index referring to STREAMERS
    test_messages = []
    test_labels = []
    label_index = 0
    for streamer in streamer_list:
        chat = load_streamer_chat(streamer)
        length = len(chat)
        train_messages_len = length - length//test_ratio  # Reserve the last 10 percent of each streamer chat for test

        train_chat = chat[:train_messages_len] 
        train_messages.extend(train_chat)
        test_chat = chat[train_messages_len: ]
        test_messages.extend(test_chat)

        train_labels.extend([label_index for i in range(len(train_chat))])
        test_labels.extend([label_index for i in range(len(test_chat))])

        label_index += 1

    return train_messages, train_labels, test_messages, test_labels