import csv
import json
import numpy as np
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util, InputExample, losses, models
from scipy.stats import multivariate_normal
from sklearn import metrics
from power_spherical import PowerSpherical


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
        
    return train_examples

def train_power_spherical(samples):
    d = samples.size(dim=1)
    mu = torch.tensor([1.] + [0.] * (d - 1), requires_grad=True)
    k = torch.tensor(1., requires_grad=True)
    dist = PowerSpherical(mu, k)

    def loss(x):
        return -torch.sum(dist.log_prob(x))

    learning_rate = 0.01
    n_iters = 50

    l = loss(samples)

    for epoch in range(n_iters):
        l_prev = l
        l = loss(samples)
        l.backward()

        with torch.no_grad():
            k -= learning_rate * k.grad
            mu -= learning_rate * mu.grad
            mu /= torch.norm(mu)

        k.grad.zero_()
        mu.grad.zero_()

        dist_new = PowerSpherical(mu, k)

    return mu, k


def train_model(cur_model_name):

    BATCH_SIZE = 16
    NUM_EPOCHS = 1
    NUM_WARMUP = 100
    NUM_STEPS_PER_EPOCH = (1143010 // BATCH_SIZE)+1

    print('Loading pre-trained model')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # Creating scratch model
    # print("Creating model from scratch")
    # word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("Creating input-example training data")
    train_examples= csv_to_inputexample()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size = BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    #
    # # The TUNING
    print('Training data loader')
    model.fit(train_objectives =[(train_dataloader, train_loss)], epochs = NUM_EPOCHS, warmup_steps = NUM_WARMUP)

    # Save model
    curr_dir = os.path.abspath(os.getcwd())
    save_path = os.path.join(curr_dir, 'trained_models/'+cur_model_name)
    os.mkdir(save_path)
    model.save(path=save_path, model_name=cur_model_name)

def test_model_PS(model_name, validating):
    if validating:
        path = 'data_processing/cleaned_data/valid/'
    else:
        path = 'data_processing/cleaned_data/test/'

    model = SentenceTransformer("./trained_models/" + model_name)

    CONTEXT_SIZE = 20

    encoding_dict = {}

    # Create dictionary of all message encodings
    print("Creating streamer encodings for:")
    print()
    for streamer_name in STREAMERS:
        print(streamer_name)
        # Load in streamer's validation/testing dataframe
        path2csv = path+streamer_name+'.csv'
        streamer_df = pd.read_csv(path2csv, keep_default_na=False)

        total_messages = streamer_df['text'].tolist()
        total_messages = total_messages[-500:]
        streamer_encodings = model.encode(total_messages, convert_to_tensor=True)

        # Store in dictionary
        encoding_dict[streamer_name] = streamer_encodings

    # get distribution of all messages
    print('Getting distribution from all messages')
    all_encodings = None
    for streamer in STREAMERS:
        streamer_encodings = encoding_dict[streamer_name]
        if all_encodings is not None:
            all_encodings = torch.cat((all_encodings, streamer_encodings))
        else:
            all_encodings = streamer_encodings
    mu, k = train_power_spherical(all_encodings)

    print('mu: ', mu)
    print('k: ', k)

def test_model_cos_sim(model_name, validating, verbose):
    # load trained model
    # iterate through list of streamers (outer loop)
    # get list of messages per streamer
    # generate list of indices for adverse message substitution

    path = None
    if validating:
        path = 'data_processing/cleaned_data/valid/'
    else:
        path = 'data_processing/cleaned_data/test/'
    
    model = SentenceTransformer("./trained_models/"+model_name)
    
    # Return a list of
    CONTEXT_SIZE = 20

    count_authentic = 0
    count_fake = 0
    total_score_authentic = 0
    total_score_fake = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_labels = []
    pred_labels = []

    encoding_dict = {}

    # Create dictionary of all message encodings
    if verbose:
        print("Creating streamer encodings for:")
        print()
    for streamer_name in STREAMERS:
        if verbose:
            print(streamer_name)
        # Load in streamer's validation/testing dataframe
        path2csv = path+streamer_name+'.csv'
        streamer_df = pd.read_csv(path2csv, keep_default_na=False)

        # Create encodings
        total_messages = streamer_df['text'].tolist()
        streamer_encodings = model.encode(total_messages, convert_to_tensor=True).cpu().numpy()

        # Store in dictionary
        encoding_dict[streamer_name] = streamer_encodings
    
    for streamer_ind in range(len(STREAMERS)):
        streamer_name = STREAMERS[streamer_ind]
        if verbose:
            print('Running tests on ' + streamer_name)
        streamer_encodings = encoding_dict[streamer_name]    

    
        for i in range(CONTEXT_SIZE, len(streamer_encodings)-CONTEXT_SIZE): # starts at CONTEXT_SIZE ends at len - CONTEXT_SIZE
            adversarial_bool  = np.random.rand() < 0.1 #10%
            encoded_context = np.append(streamer_encodings[i-CONTEXT_SIZE:i], streamer_encodings[i+1:i+CONTEXT_SIZE], axis=0)
            encoded_message = streamer_encodings[i]

            if adversarial_bool:
                # Pick another random streamer
                adv_streamer_ind = np.random.choice([i for i in range(len(STREAMERS)) if i!=streamer_ind]) #Picks streamer index, excluding current streamer
                adv_streamer_encodings = encoding_dict[STREAMERS[adv_streamer_ind]]
                encoded_message = adv_streamer_encodings[np.random.choice(range(0, len(adv_streamer_encodings)))]
            
            # Average context message embedding
            avg_embedding = np.mean(encoded_context, axis=0)

            cosine_score = util.cos_sim(encoded_message, avg_embedding)
            
            if cosine_score > .5:
                prediction = 1
            elif cosine_score < .5:
                prediction = 0
            else:
                prediction = np.random.choice([0, 1])
                

            if adversarial_bool:
                true_labels.append(0)
                count_fake += 1
                total_score_fake += cosine_score
                if prediction == 0: #correct guess
                    pred_labels.append(0)
                    true_negative += 1
                else:
                    pred_labels.append(1)
                    false_positive += 1
            else:
                true_labels.append(1)
                count_authentic += 1
                total_score_authentic += cosine_score
                if prediction == 1:
                    pred_labels.append(1)
                    true_positive += 1 
                else:
                    pred_labels.append(0)
                    false_negative += 1
        
    # Calculate averages
    average_score_authentic = total_score_authentic / count_authentic
    average_score_fake = total_score_fake / count_fake
    accuracy_true_negative = true_negative/count_fake
    accuracy_true_positive = true_positive/count_authentic
    total_accuracy = (true_negative+true_positive)/(count_authentic+count_fake)

    # Calculate precision/recall
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = ['False', 'True'])
    disp.plot()
    plt.savefig(model_name+'_confusion_matrix.png')

    print('Dataset info:')
    print('Total valid and invalid messages:', count_authentic, count_fake)
    print('average cosine score for authentic messages:', average_score_authentic)
    print('average cosine score for fake messages:', average_score_fake)
    print()

    print('Testing info:')
    print('Percent of accurate predictions when valid:', accuracy_true_positive)
    print('Percent of accurate predictions when invalid:', accuracy_true_negative)
    print('Total accuracy:', total_accuracy)

    print('True positive, false positive, true negative, false negative:', true_positive, false_positive, true_negative, false_negative)
    print('Precision:', precision)
    print('Recall:', recall)
    print()

    # if total_accuracy > 0.6:
    #     print('Woo Yeah Baby! Thats What Ive Been Waiting For! Thats what its all about!')
    # else:
    #     print('It was a misinput, IT WAS A MISINPUT')

def main():
    #If passing in arguments later:
    args = sys.argv[1:]


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    SEED = 0
    np.random.seed(SEED)
    #MODEL_NAME = 'default_paraphrase'
    MODELS = ['default_minilm','default_paraphrase', 'twitch_chatter_v1', 'twitch_chatter_v1_paraphrase']

    #train_model(MODEL_NAME)

    for model in MODELS:
        print("Testing model ", model)
        test_model_cos_sim(model, validating=True, verbose = False)

    
    # # Test on AdinRoss chat
    # path2csv = 'data_processing/cleaned_data/AdinRoss.csv'
    # df = pd.read_csv(path2csv, keep_default_na=False)
    # test_messages = df['text'].tolist()[5001:5100]
    # #print([s.encode('utf8') for s in test_messages])

    # # Encode messages
    # embeddings = model.encode(test_messages, convert_to_tensor=True)

    # # Test message
    # test_message = 'hello'
    # #print(test_message)
    # test_embedding = model.encode(test_message, convert_to_tensor=True)

    # embeddings = embeddings.cpu().numpy()
    # test_embedding = test_embedding.cpu().numpy()

    # sentences1 = ['hello how are you']
    # sentences2 = ['hi how are you doing', 'hello nice to meet you', 'hello what is your name']

    # embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    # embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # avg_embedding = torch.mean(embeddings2, 0)

    # cosine_scores = util.cos_sim(embeddings1, avg_embedding)
    # print(cosine_scores)

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