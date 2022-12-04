import csv
import util
import numpy as np
import pandas as pd

STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit',
'LVNDMARK', 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']

def get_words():
    pass

def main():
    list_of_dataframes = util.load_messages(STREAMERS, 30)

if __name__ == "__main__":
    main()