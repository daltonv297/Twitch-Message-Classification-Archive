import csv

import numpy as np
import pandas as pd
import json

# Returns a list of messages from a given streamer's logs
def load_streamer_chat(streamer):
    
    csv_path = 'twitch-listener/'+ streamer +'.csv'

    messages = []
    
    with open(csv_path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)

        for row in reader:
            messages.append(row[0])

    return messages[1:]  # Initial first message is just the 'text' header
