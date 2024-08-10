I have done the NLP task which revolves around word representations.

Task - 1: Word Similarity Scores.
    I have to come up with an unsupervised method to create word embeddings. 
    Data Preparation: I have combined an English stories corpus from Kaggle (\data\storied.txt), some English corpus from Kaggle (\data\stories.txt), and some scrapped data from Hugging Face. The hugging face data is short English children's stories (\data\scrapping.py, \data\scrapped_data.txt). These stories have different animals with character names. I have combined the data and preprocessed (\data\data_cleaning.ipynb) and stored them in the cleaned_data.txt file (\data\cleaned_data.txt). In preprocessing, I have removed all the irrelevant symbols and converted words into small case letters.
    Creating custom word embeddings: I have trained FastText and Word2Vec, both unsupervised algorithms that can be used to create custom embeddings for data. For both techniques, I used CBOW and Skip-gram methods. Codes are at 
