import numpy as np


def bag_of_words(sentences):
    tokenized_sentences = [sentence.lower().split() for sentence in sentences] #토큰화
    flat_words = [word for sublist in tokenized_sentences for word in sublist] #토큰들 한 리스트로 모음
    vocabulary = sorted(set(flat_words)) #중복 제거 및 정렬
    word_to_index = {word:i for i, word in enumerate(vocabulary)} #단어와 인덱스를 딕셔너리로 변형

    bag_of_words_matrix = np.zeros((len(sentences),len(vocabulary)), dtype = int)

    for i, sentence in enumerate(tokenized_sentences) :
        for word in sentence :
            if word in word_to_index :
                bag_of_words_matrix[i,word_to_index[word]] += 1

    return bag_of_words_matrix, vocabulary

corpus = ["This movie is awesome awesome",
          "I do not say is good, but neither awesome",
          "Awesome? Only a fool can say that"]
bow_matrix, vocabulary = bag_of_words(corpus)
print("Vocabulary:", vocabulary)
print("BoW:\n", bow_matrix)

