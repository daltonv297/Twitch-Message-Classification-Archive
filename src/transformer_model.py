import csv
import json
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util, InputExample, losses

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


def csv_to_inputexample():
    path2inputexamples = "./data_processing/input_example_pairs/"
    train_examples = [] # List of InputExample objects

    # Go through each streamer csv and create InputExample pairs
    for streamer in STREAMERS:
        streamer_csv = path2inputexamples + streamer + ".csv"
        df = pd.read_csv(streamer_csv)
        
        for row in df.itertuples(index=True, name='Pandas'):
            messages = json.loads(getattr(row, "sentence_pair"))
            print(messages)
            label = getattr(row, "similarity_score")
            train_examples.append(InputExample(texts=messages, label=label))
    
    # Split into train, test, valid
        
    return train_examples
    
def main():
    # test_adin()
    # test_combined()
    BATCH_SIZE = 16
    NUM_EPOCHS = 1
    NUM_WARMUP = 100

    print('Loading pre-trained model')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    train_examples= csv_to_inputexample()
    train_dataloader = torch.utils.data.Dataloader(train_examples, shuffle=True, batch_size = BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)
    
    # The TUNING
    model.fit(train_objectives =[(train_dataloader, train_loss)], epochs = NUM_EPOCHS, warmup_steps = NUM_WARMUP)

if __name__ == "__main__":
    main()