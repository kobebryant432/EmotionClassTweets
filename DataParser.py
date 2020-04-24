#Script for generating a panda dataframe from the raw data
##
import re
import emoji
import pandas as pd

def raw_to_panda(Data):
    """Generates a pandas data frame from the raw data. The pre-processing is also immediately applied.

           Parameters:
           Data (string): The path to the dataset

           Returns:
           data_frame (pandas Dataframe): The resulting data frame with three columns:
                                          1) (String) The tweet of tweet
                                          2) (List; Strings) The list of words in the tweet
                                          3) (Int) The emotion score or label of the tweet
          """
    data_frame = pd.DataFrame(columns=["TweetText", "Hash_tagwords", "Words", "Label"])
    with open(Data, 'r', encoding='utf-8') as file:
        header_line = next(file)
        for line in file:
            line = line.rstrip().split("\t")
            clean_text, hash_tagwords = clean_tweet(line[1])
            label = int(line[3][0])
            df1 = pd.DataFrame([[" ".join(clean_text), clean_text,hash_tagwords, label]], columns=["TweetText", "Words", "Hash_tagwords", "Label"])
            data_frame = data_frame.append(df1, ignore_index=True)
    return data_frame



def clean_tweet(text):
    """Processes the text of a tweet

        Parameters:
        text (string): The input string (tweet) which is to pe preprocessed

        Returns:
        list: A list of words representing the processed tweet

       """
    # convert to lower case
    text = text.lower()

    # detect and remove URL's
    reg = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|" \
          r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    text = re.sub(reg, 'URL', text)

    # detect and remove @ mentions
    text = re.sub(r"(?:\s|^)@[\w_-]+", ' MENTION', text)

    # replace & with "and"
    text = text.replace("&", ' and ')

    text = text.replace(r"\n", "")

    # remove the # sign
    hashtagwords = re.findall(r'(?:\s|^)#[\w_-]+', text)
    hashtagwords = [h.replace("#","") for h in hashtagwords]
    text = text.replace("#", "")

    # detect and remove ordinals
    text = re.sub(r"[0-9]+(?:st|nd|rd|th)", 'ORDINAL', text)

    #convert emoji
    for emoj in emoji.emoji_lis(text):
        text = text.replace(emoj['emoji'], " " + emoj['emoji'] + " ")
    text = emoji.demojize(text)

    #encoding punctuation
    text = text.replace("_", "")
    text = text.replace("'", '')
    text = text.replace("’", '')
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("?", " ? ")
    text = text.replace("!", " ! ")

    # remove all characters except a to z and apostrophes (i.e. numbers etc)
    text = re.sub(r"[^A-Za-z'’,.?_!]+", ' ', text)

    special_words = ["MENTION", "ORDINAL", "URL"]

    return [word for word in text.split() if word not in special_words], hashtagwords