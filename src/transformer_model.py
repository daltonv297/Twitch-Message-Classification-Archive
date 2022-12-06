import csv
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util, InputExample, losses, models
from scipy.stats import multivariate_normal

STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit',
 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']



def test_adin():

    # Load the SentenceTransformer model
    model = SentenceTransformer('adin_5000')

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
        df = pd.read_csv(streamer_csv, keep_default_na=False)
        
        for row in df.itertuples(index=True, name='Pandas'):
            message1 = getattr(row, 'message1')
            message2 = getattr(row, 'message2')
            label = getattr(row, "similarity_score")
            train_examples.append(InputExample(texts=[message1, message2], label=label))
    
    # Split into train, test, valid
        
    return train_examples
    
def train_model():
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print() 
    # test_adin()
    # test_combined()
    BATCH_SIZE = 16
    NUM_EPOCHS = 1
    NUM_WARMUP = 100

    print('Loading pre-trained model')
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Creating scratch model
    # print("Creating model from scratch")
    # word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("Creating inputexample training data")
    train_examples= csv_to_inputexample()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size = BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)
    #train_loss = losses.MultipleNegativeRanj

    # The TUNING
    print('Training data loader')
    model.fit(train_objectives =[(train_dataloader, train_loss)], epochs = NUM_EPOCHS, warmup_steps = NUM_WARMUP)

    # Save model
    model.save(path='./trained_models/all_data', model_name='all_data')

def main():
    #train_model()

    # load trained model
    # iterate through list of streamers (outer loop)

    model = SentenceTransformer("./trained_models/all_data")
    # Test on AdinRoss chat
    path2csv = 'data_processing/cleaned_data/AdinRoss.csv'
    df = pd.read_csv(path2csv, keep_default_na=False)
    test_messages = df['text'].tolist()[5001:5100]
    #print([s.encode('utf8') for s in test_messages])

    # Encode messages
    embeddings = model.encode(test_messages, convert_to_tensor=True)

    # Test message
    test_message = 'hello'
    #print(test_message)
    test_embedding = model.encode(test_message, convert_to_tensor=True)

    embeddings = embeddings.cpu().numpy()
    test_embedding = test_embedding.cpu().numpy()

    sentences1 = ['hello how are you']
    sentences2 = ['hi how are you doing', 'hello nice to meet you', 'hello what is your name']

    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    avg_embedding = torch.mean(embeddings2, 0)

    cosine_scores = util.cos_sim(embeddings1, avg_embedding)
    print(cosine_scores)

    # n,d = embeddings.shape
    # mean = np.mean(embeddings, axis=0)
    # sigma = np.cov(embeddings.T)
    # print(np.linalg.norm(test_embedding))
    # print(mean.shape)
    #
    # probability = multivariate_normal.pdf(test_embedding, mean, sigma)

    #print("Probability", probability)

if __name__ == "__main__":
    main()