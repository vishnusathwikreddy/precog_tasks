I have done the NLP task which revolves around word representations.

Task - 1: Word Similarity Scores.

I have to come up with an unsupervised method to create word embeddings.
    
Data Preparation: I have combined an English stories corpus from Kaggle (\data\storied.txt), some English corpus from Kaggle (\data\stories.txt), and some    scrapped data from Hugging Face. The hugging face data is short English children's stories (\data\scrapping.py, \data\scrapped_data.txt).                         These stories have different animals with character names. I have combined the data and preprocessed (\data\data_cleaning.ipynb) and stored                       them in the cleaned_data.txt file (\data\cleaned_data.txt). In preprocessing, I have removed all the irrelevant symbols and converted words                       into small case letters. Total number of tokens in the cleaned data is 398580.
    
Creating custom word embeddings: I have trained FastText and Word2Vec, both unsupervised algorithms that can be used to create custom embeddings for data. For     both techniques, I used CBOW and Skip-gram methods. Codes are at '/task_1' file. The results of the models tested on the simlex999 dataset are stored in the task_1_final_results.csv file (\task1\task_1_final_results.csv). In the results file, each row has the following contents: Word1, Word2, simlex999 score, word2vec_cbow similarity, word2vec_skipgram similarity, fasttext_cbow_similarity, fasttext_skipgram_similarity. Similarity scores are based on cosine similarity. Word2vec CBOW and word2vec skip-gram models are saved at \task1\word2vec. 

I have also tested all the models with 4 words in their top 10 nearest similarity scores. Here are the sample results.

<img width="200" alt="Screenshot 1946-05-20 at 4 46 58" src="https://github.com/user-attachments/assets/c949b002-731a-47c5-bb1e-5971c2f7a90e"> <img width="200" alt="Screenshot 1946-05-20 at 4 46 03" src="https://github.com/user-attachments/assets/3e93b116-2d56-4259-907d-d4c58442c6d4"> <img width="200" alt="Screenshot 1946-05-20 at 4 46 33" src="https://github.com/user-attachments/assets/f01b82d4-0836-40dc-b344-3d6bcb463ebc"> <img width="200" alt="Screenshot 1946-05-20 at 4 46 52" src="https://github.com/user-attachments/assets/5d92d658-c85e-4920-9785-f31322f61f28">


<img width="200" alt="Screenshot 1946-05-20 at 4 46 45" src="https://github.com/user-attachments/assets/2084e071-aea6-4a01-a1bb-870fce2e7996"> <img width="200" alt="Screenshot 1946-05-20 at 4 46 58" src="https://github.com/user-attachments/assets/4d7bae99-69a7-444e-991d-be83c1313b61"> <img width="200" alt="Screenshot 1946-05-20 at 4 46 19" src="https://github.com/user-attachments/assets/bb7d6132-1075-4ff1-9df1-9b1a5d7a3af7"> <img width="200" alt="Screenshot 1946-05-20 at 4 46 37" src="https://github.com/user-attachments/assets/26f1085a-740d-4fc9-b4c8-63e2b000df00">



Task - 2 Phrase similarity.

In this task word embeddings have to be converted into phrase embeddings, and then a binary classifier has to be built to classify them as similar(1) or not (0).

Phrases generally don't carry any much semantic meaning or context compared to sentences. So for every phrase in the dataset, stop words are removed and each word is converted into (300,) dimensional vector. For OOV words, a zero vector of (300,) dimension is assigned. There are 7004 training examples, 1000 val examples and 2000 test examples in the dataset.

For generating embeddings I have downloaded 'wiki-news-300d-1M.vec.zip'  from 'https://fasttext.cc/docs/en/english-vectors.html'. It is a pretrained  FastText model file containing word embeddings.

For converting the word embeddings into phrase embeddings, I have tried two methods. 

1. Averaging the embeddings of words in the phrase to make it into a single vector. Later the vector is flattened and passed through SVM, RandomForest, Logistic regression, K-NN models. Both validation and test accuracies are recorded.
   

<img width="942" alt="Screenshot 1946-05-20 at 4 46 41" src="https://github.com/user-attachments/assets/4ca87349-a573-4c63-a0b5-31312c1f6a55">

2. Weighted averaging the word embeddings of the phrase to make it into a single vector. Later the vector is flattened and passed through SVM, RandomForest, Logistic regression, K-NN models. Both validation and test accuracies are recorded.
weight of the word is the count of the word divided by the total number of words this includes all the phrases in train, val and test data. There is no need to worry about the usual repeating words because stop words are repeated
                Weight of word = (No of times word repeated)/(Total number of words)

                Weighted average = sum(weight of word* embedding of word)/ total number of words.


<img width="942" alt="Screenshot 1946-05-20 at 4 46 23" src="https://github.com/user-attachments/assets/2443f089-d1f3-4462-8936-9b6860ad4827">

Even after using two different techniques, I am getting the exact same results even after running the code many times.


Task-2 Sentence Similarity :

This part is started with data preprocessing. Stop words are removed first and all the unnecessary symbols are removed and the text is converted into small letters. The average number of words in each of the sentences in all the data is 21. The max length is set to 25 and all the with sentences more than 25 words are truncated and the sentences with less than 25 words are padded with the 'EXTRA_TOKEN' word. The number 25 is fixed after many trial and errors. The training, validation, and testing data are of lengths 15000, 4000, and 8000 respectively. Only 15000 training examples are taken because the colab is crashing. These numbers are fixed after several train and errors.

Now the embeddings are generated with the same model used in the previous task. for the 'EXTRA_TOKEN' token and OOV words vector with zeros is assigned. 

As the sentences carry semantic and contextual information, it is not a good idea to average embeddings. There is a need to capture the information in the sentence. So LSTM with dense layers is used and machine learning models are also used.

This is the summary of the LSTM model.
<img width="942" alt="Screenshot 1946-05-20 at 4 46 30" src="https://github.com/user-attachments/assets/e162b1ea-32af-47ba-948b-2788f81b691a">


For ML models, the inputs are first flattened and PCA is applied to reduce the dimensionality. PCA is applied because, with high dimensional tensors the colab is crashing. Then the features are passed SVM, RandomForest, Logistic regression, K-NN. The results are as follows

<img width="1128" alt="Screenshot 1946-05-20 at 4 46 09" src="https://github.com/user-attachments/assets/31e49b49-8708-478e-9b4b-217809d834fa">

I tried to increase the accuracy of the Neural network model by increasing the number of epochs and data. RAM crashes if data is increased and overfits if the epochs are increased.
I feel the results can be improved if the sentences are not truncated and the training data is increased.



