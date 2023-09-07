#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download('all-corpora')


# In[3]:


nltk.download('punkt')


# In[4]:


from nltk.tokenize import word_tokenize


# In[5]:


sentence = "Natural language processing (NLP) is a subfield of computer science, \
information engineering, and artificial intelligence concerned with the interactions \
between computers and human (natural) languages, in particular how to program computers \
to process and analyze large amounts of natural language data."

print(word_tokenize(sentence))


# In[6]:


from nltk.tokenize import sent_tokenize


# In[7]:


paragraph = "Natural language processing (NLP) is a subfield of computer science, \
information engineering, and artificial intelligence concerned with the interactions \
between computers and human (natural) languages, in particular how to program computers \
to process and analyze large amounts of natural language data. Challenges in natural \
language processing frequently involve speech recognition, natural language \
understanding, and natural language generation."

print(sent_tokenize(paragraph))


# In[8]:


from nltk.corpus import stopwords

# nltk의 불용어 사전?
stopwords.words('english')[:20]


# In[9]:


stopwords.words('english')[-20:]


# In[10]:


len(stopwords.words('english'))


# In[11]:


stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(sentence)

result = []

for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print(word_tokens)
print(result) #불용어 제거한 결과물


# In[12]:


import spacy


# In[13]:


nlp = spacy.load('en_core_web_sm')
doc = nlp(sentence)


# In[14]:


word_tokenized_sentence = [token.text for token in doc]
print(word_tokenized_sentence) #blank 단위로 잘라짐.


# In[15]:


sentence_tokenized_list = [sent.text for sent in doc.sents]
print(sentence_tokenized_list)


# In[16]:


import konlpy


# In[17]:


from konlpy.tag import Hannanum
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Okt


# In[18]:


okt = Okt()
text = "한글 자연어 처리는 재밌다 이제부터 열심히 해야지ㅎㅎㅎ"

print(okt.morphs(text)) # 한글 자연어 처리는 단순히 blank 단위로 잘라서는 안된다.


# In[19]:


print(okt.morphs(text, stem=True)) # Stem=True 옵션을 통해 "해야지" 가 "하다" 라는 동사로 바뀜


# In[21]:


print(okt.nouns(text)) # nouns는 문장에서 명사만 뽑아줌.


# In[22]:


print(okt.phrases(text)) # 어절 단위로 추출??


# In[23]:


print(okt.pos(text)) # morphs로 자른 각 결과물이 어떤 종류의 텍스트인지 분석


# In[24]:


kkma = Kkma()
print(kkma.morphs(text))
print(kkma.nouns(text))
print(kkma.pos(text))


# In[25]:


komoran = Komoran()
print(komoran.morphs(text))
print(komoran.nouns(text))
print(komoran.pos(text))


# In[26]:


hannanum = Hannanum()
print(hannanum.morphs(text))
print(hannanum.nouns(text))
print(hannanum.pos(text))


# In[ ]:




