import csv
import util
import numpy as np
import pandas as pd

STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit',
'LVNDMARK', 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']

def get_words():
    pass

def create_list_of_dataframes():
    list_of_dataframes = util.load_messages(STREAMERS, 30)
    
    # Save path
    save_path = 'cleaned_data/'

    # Save each dataframe as a csv file
    for i in range(len(list_of_dataframes)):
        streamer_name = STREAMERS[i]
        list_of_dataframes[i].to_csv(save_path + streamer_name + '.csv')

    # Save combined dataframe as a csv file
    combined_dataframe = pd.concat(list_of_dataframes)
    combined_dataframe.to_csv(save_path + 'combined_dataframe.csv')


def main():
    # Create and save streamer dataframes
    create_list_of_dataframes()

if __name__ == "__main__":
    main()