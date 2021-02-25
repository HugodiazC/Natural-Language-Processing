# -*- coding: utf-8 -*-

"""Código para realizar Procesamiento de Lenguaje Natural aplicando código de análisis de sentimientos de polaridad, de frecuencia, 
   al final realizaremos un document-term matrix para alimentar un algoritmo que pueda generar texto a partir del input"""

#importar data sets
import os
import tweepy as tw
import pandas as pd


# Collect tweets 
tweets = tw.Cursor(api.search,
              q=new_search,
              lang="en",
              since=date_since).items(500)
tweets
#Texto de los tweets
for tweet in tweets:
    print(tweet.text)

tweets = tw.Cursor(api.search,
                       q=new_search,
                       lang="en",
                       since=date_since).items(500)

[tweet.text for tweet in tweets]
text_PRINT=['regular discussions with medical staff, open to ideas.\nQ why 1 director for 3-4 hospitals.\nA we want local decision… https://t.co/heTnNIFhsT',
 'Our medical director got exposed to COVID19. Our office supervisors is COVID19 positive. I hope all the office staf… https://t.co/tHmDhgLbTH',
 '@LisaRit64806190 @maddow And exactly WHO do you propose that we staff them with? The medical community is stretched… https://t.co/kyqscz37n2',
 '@BakhtawarBZ @banbhan_raza NI♥️CD\nProvides free treatment\nWith qualified Drs and para medical staff\nMedicines durin… https://t.co/iWhsgz4HWk',
 'Why are ppl having trouble understanding what air borne transmission means?! Masks have always been used in patient… https://t.co/x8o5AY9Aj4',
 '“Many rural counties do not have the public health infrastructure to cope with #COVIDー19 outbreak –at least in part… https://t.co/bRqzFfJSFw',
 '#TurningTheCorner...into a hospital crisis. #COVID19 cases skyrocketing. ICUs overflowing. Medical staff exhaustion… https://t.co/luRfkXb45Q',
 "The Absence of Due Process in  Physician Employment: It may be that a physician's employment contract conditions em… https://t.co/cGEBPSTwKM",
 'FEO staff having some fun dressing up for Halloween. We are honouring our brothers and sisters on the medical front… https://t.co/cQgnIzsf2F',
 "@keylimereveal @eath1223 Yup. And if medical staff needed hazmat protection why did the gov't not think to provide… https://t.co/KOGFmSg0Mw",
 'Is #COVID19 more dangerous in Europe? Or are EU governments prepared with the safest PPE knowing this would occur &amp;… https://t.co/HAl6TscA34',
 '@oilman900 This comes with the usual disclaimer. Performed by a professional masochistic on a closed set with full… https://t.co/651abkwohH',
 'as cancer, heart disease etc. Furthermore, because the virus will be rife in the community, accident &amp; emergency, i… https://t.co/XhakGTFvKp',
 '@JamesMelville Lets also see the costs of not locking down: hospitals overwhelmed, people dying at home/in street(?… https://t.co/xgkqKEDC0E',
 'im still in my diagnosis journey and it SUCKS. having doctors and other medical staff like you makes the journey th… https://t.co/Vt8z7PKXvr',
 'Stumping for Trump in the Tampa heat, @GovRonDeSantis seems to be handling ATC for the medical staff. The crowd has… https://t.co/db0HVGy4es',
 '@MikeReedyFF @ProFootballTalk What a bad read. Terrible medical staff mike',
 '@realDonaldTrump How about some sympathy for American medical staff, families of people who are sick with, have die… https://t.co/ffYDsXMExL',
 'Staff members of the medical records department at the former hospital site were known for their creative Halloween… https://t.co/JADaiYWtL2',
 'I don’t accept any medical professionals, ancillary staff and security working at a Covid test Center as people who… https://t.co/aHYR6RScCn',
 '@rotoworld @Rotoworld_FB Fire the medical staff. Uncalled for neglegence',
 'Proud that #Inequalities podcast (incl link to #HealthInequalities) @TheKingsFund @helenamacarena @D_R_Williams1 ha… https://t.co/B4UnjWhULu',
 'The team of volunteers &amp; staff in #Beirut are still making home visits and following up on the cases we treated at… https://t.co/G5APqotR4H',
 '@politico  Politico attack on Trump vs Biden Pandemic Response: Trump followed the advice of Fauci at the beginning… https://t.co/s6B3lH1uP9',
 'Your doctors should be referred to as your "white coats" not "stained coats." Unitex can deliver the high-quality a… https://t.co/S1Gg54MtCt',
 '"According to Dr. Richard Feifer, Chief Medical Officer at the New Martinsville Center, 47 residents and 33 staff m… https://t.co/w7TQpHJVik',
 '@JDSHELBURNE It’s medical office. I totally understand your anger...totally. But having worked in these settings th… https://t.co/QNhlcG5XHb',
 'This is what medical students have fed back.\n\nThis is unacceptable.\n\nZero tolerance means zero tolerance.\n\nWe need… https://t.co/ugz7FUqgxz',
 'An early Happy Halloween from the providers and staff of the Adult Urology Clinic @umassmemorial medical center.  M… https://t.co/FmWXPWnl06',
 '@Pedro_oi812 @edwardtrem @JulieWy53975428 @VictorianCHO Doctors, dentists, medical staff, etc have been wearing mas… https://t.co/wBtwbcIkvj',
 '@eoinyk @PMc276 @DonnellyStephen 1/. I, too, have ideas.\n\nBut unless I have practical strategies to implement them… https://t.co/nElaGwUtpd',
 'Dr. Bram Rochwerg is a medical leader in the intensive care unit at Juravinski Hospital. In April, he started hosti… https://t.co/8AogB01scQ',
 'Today is one of those days where I ask myself why I chose to work with humans. 🙃😑\n\nI really am a people person but… https://t.co/4FdbnrmTxb',
 "We have a decent set of players rn (save a few), the youth brimming with very good talent, all that's left is a man… https://t.co/0i6kFBbRba",
 '@DrOsamaSiddique @abbasnasir59 What exectly he did in Pandemic.🤔🤔🤔🤔👎\nI think Drs. Nurses, Pera medical staff deserve all respect',
 '@T1na1201 @SkyNewsBreak The deaths are on the low side because its only people who have died within 24 days of a te… https://t.co/YMSbhg6qJS',
 '@TeamTrump @realDonaldTrump TmTrp: "The hospitality industry can count on me to get them back" REALLY? He put the h… https://t.co/nRrjAi1mq4',
 '@brynbehr @boymomed The @HarrisVotes Twitter account is spectacular. One of the 24-hour sites is the Houston Med Ce… https://t.co/zEexVzZAzy',
 '@Ascher_Society Yes, absolutely true - highest rist in the geriatric wards. But together with the medical staff we… https://t.co/3qDJfJnJpo',
 'Paramedic, Nurse - LPN, LVN or RN, Naturopath, Chiropractor, OR Physician Assistant (Medical Staff Associate): Unit… https://t.co/iH9MYW9PSU',
 'Just had a con with a medical expert. He was self-isolating after having tested COVID +ve. He was asymptomatic. It… https://t.co/mtvpDZT5Ma',
 '“I tell people all the time, getting injured was horrible, but I was one of the luckiest people in the world to get… https://t.co/W9LnnhPH7R',
 '@AngelaB17229061 Unless the staff is quarantining with the residents...its going to happen. At least they called me… https://t.co/UkSAbWtxbl',
 "Health's Medical Assistant program. She also serves as Adjunct Faculty for West Coast University's BSN program and… https://t.co/DJu8crnG4p",
 '@AdamSchefter Fire the whole coaching staff and medical team',
 'There’s still time to sign up for CougarCare &amp; be entered to win an Amazon gift card. Ends Nov. 14. Check Collin em… https://t.co/Wuvhh5F6Jz',
 '#ICYMI Milliken donates more than 23,000 reusable medical-grade gowns to school districts across Upstate South Caro… https://t.co/PGAMhDvMU3',
 '@kirstiealley I\'m some states certain family members are being considered "Essential caretakers,"thus being allowed… https://t.co/UnbPhhun7C',
 '@bonnocke @Rob_Sisson @March_for_Life If medical staff decided to quit and no one could be hired in their place, co… https://t.co/ThCkj0xLGt',
 'Do not go to Einstein Medical Center in Norristown. \n\nThey fucked up a prescription 3-4x. Please try another hospit… https://t.co/KxZOuHMH0z']
    
 #Quien tweetea y usuarios   
tweets_2 = tw.Cursor(api.search, 
                           q=new_search,
                           lang="en",
                           since=date_since).items(500)

users_locs = [[tweet.user.screen_name, tweet.user.location] for tweet in tweets_2]
users_locs
  
#Acomodar en un data set de Pandas
tweet_text = pd.DataFrame(data=users_locs, 
                    columns=['user', "location"])
tweet_text
tweet_text['Tweets']=text_PRINT

#Cleaning Data
import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda text_PRINT: clean_text_round1(text_PRINT)

def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda text_PRINT: clean_text_round2(text_PRINT)

tweet_text['Tweets']=text_PRINT
""" limpieza data SET
import re 
import nltk
nltk.download('stopwords') #extraer plabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #es para volver todo infinitivo
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ', tweet_text['Tweets'][i]) #quitar signos puntuación 
    review = review.lower() #pasar a minúsculas todo
    review = review.split() #separar las palabras
    ps= PorterStemmer()
    review = [ps.stem(word) for word in Tweets if not word in set(stopwords.words('english'))] 
    review = ' '.join(review) #unir todo en palabras separdas por un espacio en blanco
    corpus.append(Tweets)"""

#Pickle the text
import pickle
text_PRINT.to_pickle("corpus.pkl")
#Contar palabras 
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(text_PRINT)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = text_PRINT.index
data_dtm

# EMPIEZA DE NUEVO TODO EL CODIGO  *      *        *


Text_TOKEN= """'regular discussions with medical staff, open to ideas.\nQ why 1 director for 3-4 hospitals.\nA we want local decision… https://t.co/heTnNIFhsT',
 'Our medical director got exposed to COVID19. Our office supervisors is COVID19 positive. I hope all the office staf… https://t.co/tHmDhgLbTH',
 '@LisaRit64806190 @maddow And exactly WHO do you propose that we staff them with? The medical community is stretched… https://t.co/kyqscz37n2',
 '@BakhtawarBZ @banbhan_raza NI♥️CD\nProvides free treatment\nWith qualified Drs and para medical staff\nMedicines durin… https://t.co/iWhsgz4HWk',
 'Why are ppl having trouble understanding what air borne transmission means?! Masks have always been used in patient… https://t.co/x8o5AY9Aj4',
 '“Many rural counties do not have the public health infrastructure to cope with #COVIDー19 outbreak –at least in part… https://t.co/bRqzFfJSFw',
 '#TurningTheCorner...into a hospital crisis. #COVID19 cases skyrocketing. ICUs overflowing. Medical staff exhaustion… https://t.co/luRfkXb45Q',
 "The Absence of Due Process in  Physician Employment: It may be that a physician's employment contract conditions em… https://t.co/cGEBPSTwKM",
 'FEO staff having some fun dressing up for Halloween. We are honouring our brothers and sisters on the medical front… https://t.co/cQgnIzsf2F',
 "@keylimereveal @eath1223 Yup. And if medical staff needed hazmat protection why did the gov't not think to provide… https://t.co/KOGFmSg0Mw",
 'Is #COVID19 more dangerous in Europe? Or are EU governments prepared with the safest PPE knowing this would occur &amp;… https://t.co/HAl6TscA34',
 '@oilman900 This comes with the usual disclaimer. Performed by a professional masochistic on a closed set with full… https://t.co/651abkwohH',
 'as cancer, heart disease etc. Furthermore, because the virus will be rife in the community, accident &amp; emergency, i… https://t.co/XhakGTFvKp',
 '@JamesMelville Lets also see the costs of not locking down: hospitals overwhelmed, people dying at home/in street(?… https://t.co/xgkqKEDC0E',
 'im still in my diagnosis journey and it SUCKS. having doctors and other medical staff like you makes the journey th… https://t.co/Vt8z7PKXvr',
 'Stumping for Trump in the Tampa heat, @GovRonDeSantis seems to be handling ATC for the medical staff. The crowd has… https://t.co/db0HVGy4es',
 '@MikeReedyFF @ProFootballTalk What a bad read. Terrible medical staff mike',
 '@realDonaldTrump How about some sympathy for American medical staff, families of people who are sick with, have die… https://t.co/ffYDsXMExL',
 'Staff members of the medical records department at the former hospital site were known for their creative Halloween… https://t.co/JADaiYWtL2',
 'I don’t accept any medical professionals, ancillary staff and security working at a Covid test Center as people who… https://t.co/aHYR6RScCn',
 '@rotoworld @Rotoworld_FB Fire the medical staff. Uncalled for neglegence',
 'Proud that #Inequalities podcast (incl link to #HealthInequalities) @TheKingsFund @helenamacarena @D_R_Williams1 ha… https://t.co/B4UnjWhULu',
 'The team of volunteers &amp; staff in #Beirut are still making home visits and following up on the cases we treated at… https://t.co/G5APqotR4H',
 '@politico  Politico attack on Trump vs Biden Pandemic Response: Trump followed the advice of Fauci at the beginning… https://t.co/s6B3lH1uP9',
 'Your doctors should be referred to as your "white coats" not "stained coats." Unitex can deliver the high-quality a… https://t.co/S1Gg54MtCt',
 '"According to Dr. Richard Feifer, Chief Medical Officer at the New Martinsville Center, 47 residents and 33 staff m… https://t.co/w7TQpHJVik',
 '@JDSHELBURNE It’s medical office. I totally understand your anger...totally. But having worked in these settings th… https://t.co/QNhlcG5XHb',
 'This is what medical students have fed back.\n\nThis is unacceptable.\n\nZero tolerance means zero tolerance.\n\nWe need… https://t.co/ugz7FUqgxz',
 'An early Happy Halloween from the providers and staff of the Adult Urology Clinic @umassmemorial medical center.  M… https://t.co/FmWXPWnl06',
 '@Pedro_oi812 @edwardtrem @JulieWy53975428 @VictorianCHO Doctors, dentists, medical staff, etc have been wearing mas… https://t.co/wBtwbcIkvj',
 '@eoinyk @PMc276 @DonnellyStephen 1/. I, too, have ideas.\n\nBut unless I have practical strategies to implement them… https://t.co/nElaGwUtpd',
 'Dr. Bram Rochwerg is a medical leader in the intensive care unit at Juravinski Hospital. In April, he started hosti… https://t.co/8AogB01scQ',
 'Today is one of those days where I ask myself why I chose to work with humans. 🙃😑\n\nI really am a people person but… https://t.co/4FdbnrmTxb',
 "We have a decent set of players rn (save a few), the youth brimming with very good talent, all that's left is a man… https://t.co/0i6kFBbRba",
 '@DrOsamaSiddique @abbasnasir59 What exectly he did in Pandemic.🤔🤔🤔🤔👎\nI think Drs. Nurses, Pera medical staff deserve all respect',
 '@T1na1201 @SkyNewsBreak The deaths are on the low side because its only people who have died within 24 days of a te… https://t.co/YMSbhg6qJS',
 '@TeamTrump @realDonaldTrump TmTrp: "The hospitality industry can count on me to get them back" REALLY? He put the h… https://t.co/nRrjAi1mq4',
 '@brynbehr @boymomed The @HarrisVotes Twitter account is spectacular. One of the 24-hour sites is the Houston Med Ce… https://t.co/zEexVzZAzy',
 '@Ascher_Society Yes, absolutely true - highest rist in the geriatric wards. But together with the medical staff we… https://t.co/3qDJfJnJpo',
 'Paramedic, Nurse - LPN, LVN or RN, Naturopath, Chiropractor, OR Physician Assistant (Medical Staff Associate): Unit… https://t.co/iH9MYW9PSU',
 'Just had a con with a medical expert. He was self-isolating after having tested COVID +ve. He was asymptomatic. It… https://t.co/mtvpDZT5Ma',
 '“I tell people all the time, getting injured was horrible, but I was one of the luckiest people in the world to get… https://t.co/W9LnnhPH7R',
 '@AngelaB17229061 Unless the staff is quarantining with the residents...its going to happen. At least they called me… https://t.co/UkSAbWtxbl',
 "Health's Medical Assistant program. She also serves as Adjunct Faculty for West Coast University's BSN program and… https://t.co/DJu8crnG4p",
 '@AdamSchefter Fire the whole coaching staff and medical team',
 'There’s still time to sign up for CougarCare &amp; be entered to win an Amazon gift card. Ends Nov. 14. Check Collin em… https://t.co/Wuvhh5F6Jz',
 '#ICYMI Milliken donates more than 23,000 reusable medical-grade gowns to school districts across Upstate South Caro… https://t.co/PGAMhDvMU3',
 '@kirstiealley I\'m some states certain family members are being considered "Essential caretakers,"thus being allowed… https://t.co/UnbPhhun7C',
 '@bonnocke @Rob_Sisson @March_for_Life If medical staff decided to quit and no one could be hired in their place, co… https://t.co/ThCkj0xLGt',
 'Do not go to Einstein Medical Center in Norristown. \n\nThey fucked up a prescription 3-4x. Please try another hospit… https://t.co/KxZOuHMH0z'"""

"""
#Limpiar DataSET
import re
import string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

round1 = lambda Text_TOKEN: clean_text_round1(Text_TOKEN)"""

TEXT_LOWER = Text_TOKEN.lower()
#Limpiar Data SET
import re 
import nltk
nltk.download('stopwords') #extraer plabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #es para volver todo infinitivo
ps= PorterStemmer()
CLEANDATA = [ps.stem(word) for word in TEXT_LOWER if not word in set(stopwords.words('english'))]   
"""def clean_text_round1(text):    #Ya estaba definido por eso lo comento. 
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text"""
CLEAN_DATA2=clean_text_round1(TEXT_LOWER)

#Tokenize el data set para poderlo manipular. 
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
dataset_tokenize= word_tokenize(CLEAN_DATA2)
len(CLEAN_DATA2)
users_locs_T= word_tokenize(users_locs)


#Contar frecuencia texto 
from nltk.probability import FreqDist
fdist=FreqDist()
for word in dataset_tokenize:
    fdist[word.lower()]+=1
fdist

fdist_TEXT_20=fdist.most_common(50)
fdist_TEXT_20
 
tweet_text['CleanTweets']=CLEAN_DATA2

import pickle
CLEAN_DATA2.to_pickle('data_clean.pkl')

data_clean_pd = pd.DataFrame(tweet_text)

#document-term matrix y Frecuancy Analisys 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
def create_document_term_matrix (message_list, vectorizer):
    doc_term_matrix = vectorizer.fit_transform(message_list)
    return DataFrame(doc_term_matrix.toarray(), columns=vectorizer.get_feature_names())
cv=CountVectorizer()
create_document_term_matrix(text_PRINT, cv)

Tf=TfidfVectorizer()
create_document_term_matrix(text_PRINT, Tf)

#Sentimental Analysis
from textblob import TextBlob

pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

tweet_text['polarity'] = tweet_text['Tweets'].apply(pol)
tweet_text['subjectivity'] = tweet_text['Tweets'].apply(sub)
tweet_text

#Visualización Polaridad
import matplotlib.pyplot as plt
x=tweet_text.iloc[:,3].values #Se uso [:,1:2] para convertir en matris, de dejarlos como estaba
# [:,1] hubiera sido un vector. [:,1:2] no toma en cuenta la columna 2
y=tweet_text.iloc[:,4].values

plt.scatter(x, y, color='blue')
plt.text(x+.001, y+.001, tweet_text['user'])
plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis')
plt.xlabel('<-- Negative -------- Positive -->')
plt.ylabel('<-- Facts -------- Opinions -->')
plt.show()

#Segundo intento de visualización. 
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 8]

for index in enumerate(tweet_text):
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, tweet_text['user'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


