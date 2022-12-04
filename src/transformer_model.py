import csv
import data_processing.util as util
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit',
'LVNDMARK', 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']

def main():
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test on AdinRoss chat
    path2csv = 'data_processing\cleaned_data\AdinRoss.csv'
    df = pd.read_csv(path2csv)
    messages = df['text'].tolist()

    # Use first 100 messages
    messages = messages[:100]

    print(messages)

    # Encode messages
    
    embeddings = model.encode(messages)

    # for sentence, embedding in zip(messages, embeddings):
    #     print("Sentence:", sentence)
    #     print("Embedding:", embedding)
    #     print("")




if __name__ == "__main__":
    main()