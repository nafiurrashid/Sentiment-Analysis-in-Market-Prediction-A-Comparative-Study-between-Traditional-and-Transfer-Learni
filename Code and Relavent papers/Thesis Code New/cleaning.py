""" import library """
import pandas as pd

from pprint import pprint
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import re


#cols = ['sentiment','id','date','query_string','user','text']
cols = ['A','sentiment','text']
df = pd.read_csv("neutral_tweet.csv",header=None, names=cols ,encoding='latin-1',engine='python')


df.drop(['A'],axis=1,inplace=True)

df.info()

#df.loc[df["sentiment"]=="1","sentiment"]=0
df.loc[df["sentiment"]=="2","sentiment"]=2
#df.loc[df["sentiment"]=="3","sentiment"]=2
#df.loc[df["sentiment"]=="4","sentiment"]=4
#df.loc[df["sentiment"]=="5","sentiment"]=4
df.sentiment.value_counts()
df.head(10)

df = df.dropna()
df.isnull().sum()
df['pre_clean_len'] = [len(t) for t in df.text]

data_dict = {
    'sentiment':{
        'type':df.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df.shape
}
pprint(data_dict)

fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df.pre_clean_len)
plt.show()

""" Data Cleaning """
tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')


def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()


testing = df.text[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
test_result

""" Clean the data """
nums = [1,33362]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[1]):
    
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] ) )                                                                   
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

"""print ("Cleaning and parsing the tweets...\n")
for i in range(nums[1],nums[2]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[2] ))                                                                    
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

print ("Cleaning and parsing the tweets...\n")
for i in range(nums[2],nums[3]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[3] ))                                                                    
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))
    
"""
len(clean_tweet_texts)

""" Save the clean data as a csv """
clean_df1 = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df1['target'] = df.sentiment
clean_df1.head()
t=clean_df1.dropna()
t.isnull().sum()
t = t[t.target!='not_relevant']
t = t[t.target!='sentiment']
t.to_csv('clean_tweet_neutral.csv',encoding='utf-8')
csv = 'clean_tweet_neutral.csv'
my_df1 = pd.read_csv(csv,index_col=0)
my_df1.head(10)
my_df1.target.value_counts()