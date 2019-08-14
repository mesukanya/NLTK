import nltk
nltk.download()
# from nltk.corpus import state_union
# from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer


lammatizer=WordNetLemmatizer()
print(lammatizer.lemmatize("better"))

# sample_text=state_union.raw("/home/admin1/Desktop/NLTK/testdoc.txt")
#
# custome_set_tokenizer=PunktSentenceTokenizer(sample_text)
# tokenized=custome_set_tokenizer.tokenize(sample_text)
#
#
# def process_content():
#     try:
#         for i in tokenized:
#             words=nltk.word_tokenize(i)
#             tagged=nltk.pos_tag(words)
#             print(tagged)
#
#     except Exception as e:
#

