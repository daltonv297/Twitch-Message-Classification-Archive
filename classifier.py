from venv import create
import util

streamers = ['loltyler1', 'moistcr1tikal']
STRMR_TO_CHAT = {} # Dictionary mapping streamers to list of their chat



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

def main():
    for streamer in streamers:
        chat = util.load_streamer_chat(streamer)
        STRMR_TO_CHAT[streamer] = chat
    
    print(create_dictionary(STRMR_TO_CHAT['loltyler1']))

if __name__ == "__main__":
    main()