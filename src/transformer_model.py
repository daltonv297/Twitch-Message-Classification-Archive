import csv
import json
import numpy as np
import pandas as pd
import torch
import os
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

import datetime
import string


from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util, InputExample, losses, models
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from power_spherical import PowerSpherical


STREAMERS = ['AdinRoss', 'Alinity', 'Amouranth', 'HasanAbi', 'Jerma985', 'KaiCenat', 'LIRIK', 'loltyler1', 'Loserfruit',
 'moistcr1tikal', 'NICKMERCS', 'Pestily', 'pokimane', 'shroud', 'sodapoppin', 'summit1g', 'tarik',
'Tfue', 'Wirtual', 'xQc']


def csv_to_inputexample():
    path2inputexamples = "./data_processing/input_example_pairs_chrono/"
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

def train_power_spherical(samples, learning_rate, eps,  mu_init=None, k_init=None):
    d = samples.size(dim=1)
    # d = 300
    # if torch.cuda.is_available():
    #     use_device = 'cuda'
    # else:
    #     use_device = 'cpu'
    use_device = 'cpu'

    if mu_init is None:
        mu = torch.tensor([1.] + [0.] * (d - 1), requires_grad=True, device=torch.device(use_device))
    else:
        mu = torch.tensor(mu_init.cpu().tolist(), requires_grad=True, device=torch.device(use_device))

    if k_init is None:
        k = torch.tensor(1., requires_grad=True, device=torch.device(use_device))
    else:
        k = torch.tensor(k_init.cpu().item(), requires_grad=True, device=torch.device(use_device))



    dist = PowerSpherical(mu, k)

    def loss(x):
        return -torch.sum(dist.log_prob(x), dtype=torch.float32)

    l_prev = None
    l = None
    n_iter = 0
    n_iters = 100

    while l_prev is None or abs(l.item() - l_prev.item()) >= eps:
    #for epoch in range(n_iters):
        l_prev = l
        l = loss(samples.cpu())
        l.backward()

        with torch.no_grad():
            k -= learning_rate * k.grad
            mu -= learning_rate * mu.grad
            mu /= torch.norm(mu)

        k.grad.zero_()
        mu.grad.zero_()

        dist = PowerSpherical(mu, k)

        # if l_prev is not None:
        #     print('iter ', n_iter, 'k = ', k, 'loss=', l.item(), 'loss diff= ', abs(l.item() - l_prev.item()))

        n_iter += 1

    #print('done after ', n_iter, ' iterations')

    return mu, k


def train_model(cur_model_name):

    BATCH_SIZE = 16
    NUM_EPOCHS = 1
    NUM_WARMUP = 100
    NUM_STEPS_PER_EPOCH = (1143010 // BATCH_SIZE)+1

    print('Loading pre-trained model')
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    # Creating scratch model
    # print("Creating model from scratch")
    # word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("Creating input-example training data")
    train_examples= csv_to_inputexample()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size = BATCH_SIZE)
    train_loss = losses.CosineSimilarityLoss(model)

    # The TUNING
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

    context_size_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    learning_rate = 0.001
    eps = 0.5
    prob_fake_prior = 0.5

    count_correct_real = 0
    count_real = 0
    count_correct_fake = 0
    count_fake = 0

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
        total_messages = total_messages[-2000:]
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
    mu_all, k_all = train_power_spherical(all_encodings, learning_rate=learning_rate, eps=eps)
    dist_all = PowerSpherical(mu_all, k_all)

    mu_prev = mu_all
    k_prev = k_all

    # run tests
    with open('performance_metrics/out_logs_PS_miniLM.txt', 'w') as f:
        for CONTEXT_SIZE in context_size_list:
            print('using context size ', CONTEXT_SIZE)
            for streamer_ind in range(len(STREAMERS)):
                streamer_name = STREAMERS[streamer_ind]
                print('Running tests on ' + streamer_name)
                streamer_encodings = encoding_dict[streamer_name]

                for i in range(CONTEXT_SIZE, len(streamer_encodings) - CONTEXT_SIZE):  # starts at CONTEXT_SIZE ends at len - CONTEXT_SIZE
                    # if i % 100 == 0:
                        #print('predicting message ', i, '/', len(streamer_encodings) - 2*CONTEXT_SIZE)

                    adversarial_bool = np.random.rand() < prob_fake_prior
                    encoded_context = torch.cat((streamer_encodings[i - CONTEXT_SIZE:i], streamer_encodings[i + 1:i + CONTEXT_SIZE]))
                    encoded_message = streamer_encodings[i]

                    if adversarial_bool:
                        # Pick another random streamer
                        adv_streamer_ind = np.random.choice([i for i in range(len(STREAMERS)) if i!=streamer_ind]) #Picks streamer index, excluding current streamer
                        adv_streamer_encodings = encoding_dict[STREAMERS[adv_streamer_ind]]
                        encoded_message = adv_streamer_encodings[np.random.choice(range(0, len(adv_streamer_encodings)))]

                    # train on context
                    mu, k = train_power_spherical(encoded_context, learning_rate=learning_rate, eps=eps, mu_init=mu_prev, k_init=k_prev)
                    mu_prev = mu
                    k_prev = k
                    dist_context = PowerSpherical(mu, k)
                    log_prob_given_real = dist_context.log_prob(encoded_message.cpu())
                    log_prob_given_fake = dist_all.log_prob(encoded_message.cpu())
                    m = torch.max(torch.tensor((log_prob_given_real, log_prob_given_fake)))

                    prob_real = torch.exp(log_prob_given_real - m) * (1-prob_fake_prior) / (torch.exp(log_prob_given_real - m) * (1-prob_fake_prior) + torch.exp(log_prob_given_fake - m) * prob_fake_prior)
                    if prob_real > 0.5:
                        prediction = 1
                    else:
                        prediction = 0

                    if adversarial_bool:
                        count_fake += 1
                        if prediction == 0:
                            count_correct_fake += 1
                    else:
                        count_real += 1
                        if prediction == 1:
                            count_correct_real += 1


            f.write('window size: ' + str(CONTEXT_SIZE) + '\n')
            f.write('accuracy detecting fake messages: ' + str(count_correct_fake / count_fake) + '\n')
            f.write('accuracy detecting real messages: ' + str(count_correct_real / count_real) + '\n')
            f.write('total accuracy: ' + str((count_correct_fake + count_correct_real) / (count_fake + count_real)) + '\n\n')

def test_model_cos_sim(model_name, validating, verbose, timestamp_context=5, cos_score_threshold=0.65, adversarial_ratio = 0.5, cw = 20):
    # load trained model
    # iterate through list of streamers (outer loop)
    # get list of messages per streamer
    # generate list of indices for adverse message substitution

    path = None
    if validating:
        path = 'data_processing/cleaned_data/valid_chrono/'
    else:
        path = 'data_processing/cleaned_data/test_chrono/'
    
    model = SentenceTransformer("./trained_models/"+model_name)

    CONTEXT_SIZE = cw
    CONTEXT_DURATION = timestamp_context #seconds
    SCORE_THRESHOLD = cos_score_threshold
    ADVERSARIAL_RATIO = adversarial_ratio

    # Performance tracking:
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

    # Create dictionary of all message encodings
    encoding_dict = {}
    timestamp_dict = {}

    if verbose:
        print("Creating streamer encodings for:")
        print()

    # Per streamer, create encodings
    for streamer_name in STREAMERS:
        if verbose:
            print(streamer_name)

        # Load in streamer's validation/testing dataframe
        path2csv = path+streamer_name+'.csv'
        streamer_df = pd.read_csv(path2csv, keep_default_na=False)
        streamer_df = streamer_df[:len(streamer_df)//2] # Reduce validation size by 2

        # Create encodings
        total_messages = streamer_df['text'].tolist()
        streamer_encodings = model.encode(total_messages, convert_to_tensor=True).cpu().numpy()
        # Store in dictionary
        encoding_dict[streamer_name] = streamer_encodings

        # Used when using the timestamp metric for context
        timestamp_string = streamer_df['timestamp'].tolist()
        timestamps = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') for x in timestamp_string] # List of timestamp objects
        timestamp_dict[streamer_name] = timestamps

    # Per streamer, test
    for streamer_ind in range(len(STREAMERS)):
        streamer_name = STREAMERS[streamer_ind]
        if verbose:
            print('Running tests on ' + streamer_name)
        streamer_encodings = encoding_dict[streamer_name]    
        streamer_timestamps = timestamp_dict[streamer_name]

        for i in range(CONTEXT_SIZE, len(streamer_encodings)-CONTEXT_SIZE): # starts at CONTEXT_SIZE ends at len - CONTEXT_SIZE
            adversarial_bool  = np.random.rand() < ADVERSARIAL_RATIO #50%
            encoded_message = streamer_encodings[i]
            encoded_context = []

            # If context window based on timestamp
            if timestamp_context:
                cur_ts = streamer_timestamps[i]
                reached_prev_limit = False
                reached_post_limit = False
                prev_limit_index = i-1
                post_limit_index = i+1
                # Find the earliest message to include in context
                while not reached_prev_limit and prev_limit_index >= 0:
                    prev_message_ts = streamer_timestamps[prev_limit_index]
                    #print((cur_ts - prev_message_ts).seconds)
                    if (cur_ts - prev_message_ts).seconds < CONTEXT_DURATION:
                        prev_limit_index -= 1
                    else:
                        reached_prev_limit = True
                # Find latest message to include in context
                while not reached_post_limit and post_limit_index < len(streamer_encodings):
                    post_message_ts = streamer_timestamps[post_limit_index]
                    #print((post_message_ts - cur_ts).seconds)
                    if (post_message_ts - cur_ts).seconds < CONTEXT_DURATION:
                        post_limit_index += 1
                    else:
                        reached_post_limit = True

                #print('Context window of ', CONTEXT_DURATION, ' seconds had ', post_limit_index-prev_limit_index, ' messages')
                encoded_context = np.append(streamer_encodings[prev_limit_index:i], streamer_encodings[i+1:post_limit_index], axis=0)

            # If context window based on number of messages
            else:
                encoded_context = np.append(streamer_encodings[i-CONTEXT_SIZE:i], streamer_encodings[i+1:i+CONTEXT_SIZE], axis=0)

            # Insert adversarial streamer message
            if adversarial_bool:
                # Pick another random streamer
                adv_streamer_ind = np.random.choice([i for i in range(len(STREAMERS)) if i!=streamer_ind]) #Picks streamer index, excluding current streamer
                adv_streamer_encodings = encoding_dict[STREAMERS[adv_streamer_ind]]
                encoded_message = adv_streamer_encodings[np.random.choice(range(0, len(adv_streamer_encodings)))]
            
            # Average context message embedding
            avg_embedding = np.mean(encoded_context, axis=0)

            cosine_score = util.cos_sim(encoded_message, avg_embedding)
            
            if cosine_score >= SCORE_THRESHOLD:
                prediction = 1
            elif cosine_score < SCORE_THRESHOLD:
                prediction = 0
            # else:
            #     prediction = np.random.choice([0, 1])
                

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
    confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels, normalize = 'true')
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = ['False', 'True'])
    disp.plot()
    if timestamp_context:
        plt.savefig('performance_metrics/confusion_matrices/'+model_name+'_CD='+str(CONTEXT_DURATION)+'_P='+str(SCORE_THRESHOLD)+'_AR='+str(ADVERSARIAL_RATIO)+'.png')
    else:
        plt.savefig('performance_metrics/confusion_matrices/'+model_name+'_CS='+str(CONTEXT_SIZE)+'_P='+str(SCORE_THRESHOLD)+'_AR='+str(ADVERSARIAL_RATIO)+'.png')

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
    return total_accuracy

def visualize_embeddings(model_name):

    path = 'data_processing/cleaned_data/test/'

    model = SentenceTransformer("./trained_models/" + model_name)
    encoding_dict = {}

    num_samples = 500

    # Create dictionary of all message encodings
    print("Creating streamer encodings for:")
    print()
    for streamer_name in STREAMERS:
        print(streamer_name)
        # Load in streamer's validation/testing dataframe
        path2csv = path+streamer_name+'.csv'
        streamer_df = pd.read_csv(path2csv, keep_default_na=False)

        total_messages = streamer_df['text'].tolist()
        if len(total_messages) > num_samples:
            total_messages = np.random.choice(total_messages, size=num_samples, replace=False)
        streamer_encodings = model.encode(total_messages, convert_to_tensor=True)

        # Store in dictionary
        encoding_dict[streamer_name] = streamer_encodings

    all_encodings = None
    for streamer_name in STREAMERS:
        streamer_encodings = encoding_dict[streamer_name]
        if all_encodings is not None:
            all_encodings = torch.cat((all_encodings, streamer_encodings))
        else:
            all_encodings = streamer_encodings

    #all_encodings = all_encodings.numpy()


    # svd = TruncatedSVD(n_components=k, n_iter=n_iter)
    # svd.fit(all_encodings.cpu())
    # embeddings_reduced = svd.transform(all_encodings.cpu())

    pca3 = PCA(n_components=3)
    pca3.fit(all_encodings.cpu())


    theta, phi = np.linspace(0, 2 * np.pi, 20), np.linspace(0, np.pi, 20)
    THETA, PHI = np.meshgrid(theta, phi)
    X, Y, Z = np.sin(PHI) * np.cos(THETA), np.sin(PHI) * np.sin(THETA), np.cos(PHI)

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(X, Y, Z, linewidth=1, alpha=0.25, color="gray")

    cm = plt.get_cmap('gist_ncar')
    num_colors = len(STREAMERS)
    color_list = [cm(1.*i/num_colors) for i in np.random.permutation(num_colors)]
    ax.set_prop_cycle(color=color_list)

    for streamer in STREAMERS:
        embeddings_reduced = pca3.transform(encoding_dict[streamer].cpu())
        embeddings_reduced = embeddings_reduced / np.linalg.norm(embeddings_reduced, axis=1).reshape(-1, 1)
        X, Y, Z = embeddings_reduced.T
        ax.scatter(X, Y, Z, s=5, label=streamer)

    # ax.plot(*torch.stack((torch.zeros_like(loc), loc)).T, linewidth=4, label="$\kappa={}$".format(scale))

    # X, Y, Z = embeddings_reduced.T
    # ax.scatter(X, Y, Z, s=5)

    ax.view_init(30, 45)
    ax.tick_params(axis='both')
    plt.legend(loc='upper center', ncol=5, markerscale=3)
    plt.show()

    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    ax.set_prop_cycle(color=color_list)

    pca2 = PCA(n_components=2)
    pca2.fit(all_encodings.cpu())


    for streamer in STREAMERS:
        embeddings_reduced = pca2.transform(encoding_dict[streamer].cpu())
        X, Y = embeddings_reduced.T
        ax.scatter(X, Y, s=5, label=streamer)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5, markerscale=3)
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #If passing in arguments later:
    args = sys.argv[1:]

    #For writing to file
    # stdoutOrigin = sys.stdout
    # sys.stdout = open('performance_metrics/out_logs_miniLM_v1.txt', 'w+')
    #MODELS = ['default_minilm','default_paraphrase', 'twitch_chatter_v1_paraphrase_chrono']
    # MODELS = ['twitch_chatter_v1_chrono']
    #Hyper params
    score_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    window_sizes = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    adv_ratio = 0.5
    SEED = 0
    np.random.seed(SEED)
    MODEL_NAME = 'twitch_chatter_v1_chrono'
    #MODELS = ['default_minilm','default_paraphrase', 'twitch_chatter_v1', 'twitch_chatter_v1_paraphrase']

    #train_model(MODEL_NAME)

    test_model_PS(MODEL_NAME, validating=True)

    # for model in MODELS:
    #     print("Testing model ", MODELS[0])
    #     print("Testing cosine score thresholds:")
    #
    #     # Find best score threshold
    #     cur_best_score_index = 0
    #     cur_best_accuracy_per_score = 0
    #     score_still_improving = True
    #
    #     #visualize_embeddings('twitch_chatter_v1')
    #     while score_still_improving and cur_best_score_index < len(score_thresholds):
    #         cur_score_threshold = score_thresholds[cur_best_score_index]
    #         print("Current score threshold:", cur_score_threshold)
    #         tot_accuracy = test_model_cos_sim(MODELS[0], validating = True, verbose = False, timestamp_context=False, cos_score_threshold=cur_score_threshold, adversarial_ratio=adv_ratio, cw=20)
    #         if tot_accuracy > cur_best_accuracy_per_score:
    #             cur_best_accuracy_per_score = tot_accuracy
    #             cur_best_score_index += 1
    #         else:
    #             score_still_improving = False
    #     # If we broke the loop because score stopped improving or index overflowed, then previous score was the best
    #     best_score = score_thresholds[cur_best_score_index-1]
    #     print('Best score threshold was', best_score,'with total accuracy of', cur_best_accuracy_per_score)
    #
    #     print('Testing for best context window size')
    #
    #     # Find best window context size
    #     cur_best_window_index = 0
    #     cur_best_accuracy_per_window = 0
    #     window_still_improving = True
    #
    #     while window_still_improving and cur_best_window_index < len(window_sizes):
    #         cur_window_size = window_sizes[cur_best_window_index]
    #         print('Current window size', cur_window_size)
    #         tot_accuracy = test_model_cos_sim(MODELS[0], validating = True, verbose = False, timestamp_context=False, cos_score_threshold=best_score, adversarial_ratio=adv_ratio, cw=cur_window_size)
    #         if tot_accuracy > cur_best_accuracy_per_window:
    #             cur_best_accuracy_per_window = tot_accuracy
    #             cur_best_window_index += 1
    #         else:
    #             window_still_improving = False
    #     # If we broke the loop because we didn't improve or index overflowed, then previous window size was the best
    #     best_window = window_sizes[cur_best_window_index-1]
    #     print('Best window size was', best_window, 'with total accuracy of', cur_best_accuracy_per_window)
    #
    #     print('Optimal hyperparameters of score threshold and context window size hypothesized to be approximately', best_score, 'and', best_window, 'respectively')
    #     print('Running final test on the train set')
    #     test_accuracy = test_model_cos_sim(MODELS[0], validating = False, verbose = False, timestamp_context=False, cos_score_threshold=best_score, adversarial_ratio=adv_ratio, cw=best_window)
    #
    #     print('Final total accuracy on the train set is', test_accuracy)
    #     # visualize_embeddings('twitch_chatter_v1')
    #     sys.stdout.close()
    #     sys.stdout = stdoutOrigin
if __name__ == "__main__":
    main()