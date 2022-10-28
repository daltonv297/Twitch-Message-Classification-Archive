from venv import create
import util

import numpy as np

STREAMERS = ['loltyler1', 'moistcr1tikal', 'erobb221', 'shroud', 'tarik']
TRAIN_MODEL_ON = 1 # Index in the STREAMERS list that indicates which streamer to train model on
STRMR_TO_CHAT = {} # Dictionary mapping streamers to list of their chat. Currently unused

"""
This current implementation trains a model to recognize a given streamer's messages. 
It is trained on their chat and adversarial data is fed from the chats of other streamers.
"""

def get_words(chat):
    """ 
    TODO: An important point to consider is to what degree we should strip messages
    How important is capitalization to the meaning of a twitch message?
    How does this handle emojis and emotes

    Args:
        chat: a string containing a chat message
    Returns: 
        a list of normalized words 
    """
    return chat.strip().lower().split()


def create_dictionary(chat_log):
    """ Creates a dictionary mapping words to integer indices, for a given chat log 
    Just like in PS2 spam.py implementation, we also reduce to words that appear over
    5 times

    Args:
        chat_log: a list of strings containing chat messages
    Returns:
        A python dict mapping words to integers
    """
    # Add all words to dictionary with their count
    word_count = {}
    for message in chat_log:
        word_list = get_words(message)
        for word in word_list:
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    # Add all words appearing at least 5 times to index dict
    index = 0
    word_to_index = {}
    for word, count in word_count.items():
        if count >= 5:
            word_to_index[word] = index
            index += 1

    return word_to_index



def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    num_appearances_array = np.zeros((len(messages), len(word_dictionary)))
    for i in range(len(messages)):
        word_list = get_words(messages[i])
        for word in word_list:
            if word in word_dictionary:
                j = word_dictionary[word]
                num_appearances_array[i][j] += 1

    return num_appearances_array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    # We should use multinomial event model and Laplace smoothing
    # matrix is of shape (i,j) where i is number of examples and j is size of vocab
    # matrix is also the output from transform_text
    
    n, vocab_size = matrix.shape

    d = np.sum(matrix, axis = 1) # d[i] represents the number of words in email i

    # phi_y = np.sum([labels[i] for i in range(n)])/n
    phi_y = np.sum(labels)/n
    
    # y takes on values {0,1}
    # k refers to an id of vocab word
    def phi_ky(k, y):
        numer = 1 + np.sum([matrix[i][k] for i in range(n) if labels[i] == y])
        denom = vocab_size + np.sum([d[i] for i in range(n) if labels[i] == y])
        # denom = vocab_size + np.sum(matrix[labels==y])    # This implementation improves the whole thing from 0.955 to 9.957 lol
        return numer/denom
    
    phi_y1_array = np.zeros((vocab_size, )) # phi_y1_array[i] represents p(x=word i | y=1)
    phi_y0_array = np.zeros((vocab_size, )) # phi_y0_array[i] represents p(x=word i | y=0)

    for i in range(vocab_size):
        phi_y1_array[i] = phi_ky(i, 1)
        phi_y0_array[i] = phi_ky(i, 0)
        
    return phi_y, phi_y1_array, phi_y0_array
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    # matrix is of shape (i,j) where i is number of examples and j is size of vocab
    # matrix is also the output from transform_text
    
    phi_y, phi_y1_array, phi_y0_array = model

    n, vocab_size = matrix.shape
    
    predictions = np.zeros((n, ))
    for i in range(n):
        #p_xy0 = phi_y
        #p_xy1 = phi_y
        p_xy0 = 0
        p_xy1 = 0
        for j in range(vocab_size):
            if matrix[i][j] != 0:
                p_xy0 += matrix[i][j]*np.log(phi_y0_array[j])
                p_xy1 += matrix[i][j]*np.log(phi_y1_array[j])
        if (p_xy0 > p_xy1):
            predictions[i] = 0
        else:
            predictions[i] = 1
    
    return predictions
    # *** END CODE HERE ***


def main():
    train_messages, train_labels, test_messages, test_labels = util.load_messages(STREAMERS, 10)


    train_labels = [1 if label == TRAIN_MODEL_ON else 0 for label in train_labels] # Changes labels list to be 1 for streamer of interest, 0 otherwise
    test_labels = [1 if label == TRAIN_MODEL_ON else 0 for label in test_labels]

    dictionary = create_dictionary(train_messages)
    print('Size of dictionary: ', len(dictionary))

    train_matrix = transform_text(train_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    

if __name__ == "__main__":
    main()