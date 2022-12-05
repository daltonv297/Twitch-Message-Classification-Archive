import csv
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit',
'LVNDMARK', 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']

def test_adin():

    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test on AdinRoss chat
    path2csv = 'data_processing/cleaned_data/AdinRoss.csv'
    df = pd.read_csv(path2csv, keep_default_na=False)
    messages = df['text'].tolist()[:5000]

    # Encode messages    
    embeddings = model.encode(messages, convert_to_tensor=True)

    # Test message
    test_message = 'adin6pack adin6pack adin6pack'    
    test_embedding = model.encode(test_message, convert_to_tensor=True)

    # Closest 5 messages based on cosine similarity
    cos_scores = util.cos_sim(test_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)

    print("\n\n======================\n\n")
    print("Query:", test_message)
    print("\nTop 5 most similar sentences in the data:")

    for score, idx in zip(top_results[0], top_results[1]):
        # Get channel name from message
        message_text = messages[idx]
        streamer_name = df[df['text'] == message_text]['channel_name'].tolist()[0]

        # Print message and streamer name
        print(streamer_name, ": ", message_text, "(Score: {:.4f})".format(score))

    paraphrases = util.paraphrase_mining(model, messages)

    for paraphrase in paraphrases[0:10]:
        score, i, j = paraphrase
        print("{} \t\t {} \t\t Score: {:.4f}".format(messages[i], messages[j], score))

def test_combined():
    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Test on combined chat
    path2csv = 'data_processing/cleaned_data/combined_dataframe.csv'
    df = pd.read_csv(path2csv, keep_default_na=False)
    messages = df['text'].tolist()

    # Encode messages    
    embeddings = model.encode(messages, convert_to_tensor=True)
    print("Completed message embedding...")

    # Test message
    test_message = 'pogchamp widepeepoHappy'    
    test_embedding = model.encode(test_message, convert_to_tensor=True)
    print("Completed test message embedding...")

    # Closest 5 messages based on cosine similarity
    cos_scores = util.cos_sim(test_embedding, embeddings)[0]
    top_results = torch.topk(cos_scores, k=5)
    print("Completed cosine similarity...")

    print("\n\n======================\n\n")
    print("Query:", test_message)
    print("\nTop 5 most similar sentences in the data:")

    for score, idx in zip(top_results[0], top_results[1]):
        # Get channel name from message
        message_text = messages[idx]
        streamer_name = df[df['text'] == message_text]['channel_name'].tolist()[0]

        # Print message and streamer name
        print(streamer_name, ": ", message_text, "(Score: {:.4f})".format(score))

def main():
    test_adin()
    # test_combined()


if __name__ == "__main__":
    main()