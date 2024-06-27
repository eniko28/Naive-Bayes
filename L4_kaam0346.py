import numpy as np
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

def read_file(file_name):
    line_list = []
    with open(file_name, "r") as f:
        for line in f:
            if line.strip():
                line_list.append(line.rstrip('\n'))
    return line_list

def read_directory(directory, stopwords):
    text_files = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        content = words_count(file_path, stopwords)
        text_files.append({"file_name": file_name, "content": content})
    return text_files

def words_count(file_path, stopwords):
    stop_char = [".", ",", ":", "?", "!", ")", "(", "{", "}", "[", "]", ";", "-", "subject:"]
    with open(file_path, 'r', encoding='latin-1') as file:
        words = file.read().split()
        lowercase_words = []
        for word in words:
            lowercase_words.append(word.lower())
        words = []
        frequency = []
        for word in lowercase_words:
            if word not in stopwords and word not in stop_char:
                if word in words:
                    index = words.index(word)
                    frequency[index] += 1
                else:
                    words.append(word)
                    frequency.append(1)
    result = []
    for i in range(len(words)):
        result.append({"word": words[i], "nr_of_word": frequency[i]})
    return result

def ham_spam(ham, spam, training_set, testing_set):
    ham_data = [file for file in ham if file["file_name"] in training_set]
    ham_test = [file for file in ham if file["file_name"] in testing_set]
    spam_data = [file for file in spam if file["file_name"] in training_set]
    spam_test = [file for file in spam if file["file_name"] in testing_set]
    return ham_data, ham_test, spam_data, spam_test

def nr_of_words(files):
    db = 0
    for file in files:
        for freq_word in file["content"]:
            db += freq_word["nr_of_word"]
    return db

def word_freq(ham_data, spam_data):
    ham_nr = {}
    spam_nr = {}
    s = set()
    for file in ham_data:
        for wordfreq in file["content"]:
            word = wordfreq["word"]
            freq = wordfreq["nr_of_word"]
            if word in ham_nr:
                ham_nr[word] += freq
            else:
                ham_nr[word] = freq
            s.add(word)
    for file in spam_data:
        for wordfreq in file["content"]:
            word = wordfreq["word"]
            freq = wordfreq["nr_of_word"]
            if word in spam_nr:
                spam_nr[word] += freq
            else:
                spam_nr[word] = freq
            s.add(word)
    return ham_nr, spam_nr, len(s)

def cond_prob(total_ham, total_spam, ham_word_freq, spam_word_freq, alpha, total_unique_words):
    ham_word_probabilities = {}
    spam_word_probabilities = {}
    lambda_value = 0.00000001
    for word, freq in ham_word_freq.items():
        word_probability = (freq + alpha) / (total_ham + alpha * total_unique_words)
        if word_probability < lambda_value:
            word_probability = lambda_value
        ham_word_probabilities[word] = word_probability
    for word, freq in spam_word_freq.items():
        word_probability = (freq + alpha) / (total_spam + alpha * total_unique_words)
        if word_probability < lambda_value:
            word_probability = lambda_value
        spam_word_probabilities[word] = word_probability
    return ham_word_probabilities, spam_word_probabilities

def probability_calc(mail, ham_word_probabilities, spam_word_probabilities):
    probability = 0
    for content in mail["content"]:
        word = content["word"]
        word_freq = content["nr_of_word"]
        if word in ham_word_probabilities:
            probability -= word_freq * np.log(ham_word_probabilities[word])
        if word in spam_word_probabilities:
            probability += word_freq * np.log(spam_word_probabilities[word])
    return probability

def test(ham_training, spam_training, ham_word_probabilities, spam_word_probabilities, ham_testing, spam_testing):
    ham_training_count = len(ham_training)
    spam_training_count = len(spam_training)
    p_ham = ham_training_count / (ham_training_count + spam_training_count)
    p_spam = spam_training_count / (ham_training_count + spam_training_count)
    correct = 0
    total_ham = 0
    total_spam = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    true_positive = 0

    for mail in ham_testing:
        probability = math.log(p_spam) - math.log(p_ham)
        probability += probability_calc(mail, ham_word_probabilities, spam_word_probabilities)
        if probability < 0:
            correct += 1
            true_negative += 1
        else:
            false_positive += 1
        total_ham += 1

    for mail in spam_testing:
        probability = math.log(p_spam) - math.log(p_ham)
        probability += probability_calc(mail, ham_word_probabilities, spam_word_probabilities)
        if probability > 0:
            correct += 1
            true_positive += 1
        else:
            false_negative += 1
        total_spam += 1

    total = total_spam + total_ham
    accuracy = correct / total
    return accuracy, true_positive, false_negative, false_positive, true_negative

def plot_confusion_matrix(true_positive, false_negative, false_positive, true_negative):
    confusion_matrix = np.array([[true_positive, false_negative],
                                 [false_positive, true_negative]])

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Spam', 'Ham'],
                yticklabels=['Spam', 'Ham'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def k_fold_cross_validation(ham_data, spam_data, k, alpha_values):
    random.shuffle(ham_data)
    random.shuffle(spam_data)
    total_accuracy = []
    for alpha in alpha_values:
        accuracies = []
        for i in range(k):
            fold_size = len(ham_data) // k
            ham_folds = [ham_data[j:j+fold_size] for j in range(0, len(ham_data), fold_size)]
            spam_folds = [spam_data[j:j+fold_size] for j in range(0, len(spam_data), fold_size)]

            validation_ham = ham_folds[i]
            validation_spam = spam_folds[i]
            training_ham = [item for sublist in ham_folds[:i] + ham_folds[i+1:] for item in sublist]
            training_spam = [item for sublist in spam_folds[:i] + spam_folds[i+1:] for item in sublist]

            total_ham_count = nr_of_words(training_ham)
            total_spam_count = nr_of_words(training_spam)
            ham_freq_word, spam_freq_word, total_unique_words = word_freq(training_ham, training_spam)
            ham_word_probabilities, spam_word_probabilities = cond_prob(total_ham_count, total_spam_count,
                                                                                       ham_freq_word, spam_freq_word,
                                                                                       alpha, total_unique_words)

            accuracy, _, _, _, _ = test(training_ham, training_spam, ham_word_probabilities, spam_word_probabilities,
                                  validation_ham, validation_spam)
            accuracies.append(accuracy)

        total_accuracy.append(sum(accuracies) / len(accuracies))
    return total_accuracy

def semi_supervised_learning(ham_data, spam_data, theta, alpha, stopwords):
    unlabeled_data = read_directory("ssl", stopwords)
    while True:
        total_ham_count = nr_of_words(ham_data)
        total_spam_count = nr_of_words(spam_data)
        ham_word_freq, spam_word_freq, total_unique_words = word_freq(ham_data, spam_data)
        ham_word_probabilities, spam_word_probabilities = cond_prob(total_ham_count, total_spam_count,
                                                                     ham_word_freq, spam_word_freq,
                                                                     alpha, total_unique_words)
        D2 = []
        for mail in unlabeled_data:
            probability = probability_calc(mail, ham_word_probabilities, spam_word_probabilities)
            if probability <= -math.log(theta):
                ham_data.append(mail)
                unlabeled_data.remove(mail)
            elif(probability >= math.log(theta)):
                spam_data.append(mail)
                unlabeled_data.remove(mail)
        if len(D2) == 0:
            break

    total_ham_count = nr_of_words(ham_data)
    total_spam_count = nr_of_words(spam_data)
    ham_word_freq, spam_word_freq, total_unique_words = word_freq(ham_data, spam_data)
    ham_word_probabilities, spam_word_probabilities = cond_prob(total_ham_count, total_spam_count,
                                                                ham_word_freq, spam_word_freq,
                                                                alpha, total_unique_words)

    return ham_word_probabilities, spam_word_probabilities

def main():
    training_set = read_file("train.txt")
    testing_set = read_file("test.txt")
    stopwords1 = read_file("stopwords.txt")
    stopwords2 = read_file("stopwords2.txt")
    stopwords = stopwords1 + stopwords2

    directory = "ham"
    ham = read_directory(directory, stopwords)
    directory = "spam"
    spam = read_directory(directory, stopwords)

    ham_data, ham_test, spam_data, spam_test = ham_spam(ham, spam, training_set, testing_set)

    ham_data_count = nr_of_words(ham_data)
    spam_data_count = nr_of_words(spam_data)

    ham_freq_word, spam_freq_word, s = word_freq(ham_data, spam_data)

    ham_word_probabilities, spam_word_probabilities = cond_prob(ham_data_count, spam_data_count,
                                                                               ham_freq_word, spam_freq_word, 0.01, s)
    accuracy, _, _, _, _ = test(ham_data, spam_data, ham_word_probabilities, spam_word_probabilities, ham_data, spam_data)
    print("Naive Bayes, aplha =  0.01 - train:", 1 - accuracy)

    accuracy, _, _, _, _ = test(ham_test, spam_test, ham_word_probabilities, spam_word_probabilities, ham_test, spam_test)
    print("Naive Bayes alpha = 0.01 - test:", 1 - accuracy)


    ham_word_probabilities, spam_word_probabilities = cond_prob(ham_data_count, spam_data_count,
                                                                               ham_freq_word, spam_freq_word, 0.1, s)
    accuracy, _, _, _, _ = test(ham_data, spam_data, ham_word_probabilities, spam_word_probabilities, ham_data, spam_data)
    print("Native Bayes, alpha = 0.1 - train:", 1 - accuracy)
    accuracy, _, _, _, _ = test(ham_test, spam_test, ham_word_probabilities, spam_word_probabilities, ham_test, spam_test)
    print("Native Bayes, alpha = 0.1 eset - test:", 1 - accuracy)

    ham_word_probabilities, spam_word_probabilities = cond_prob(ham_data_count, spam_data_count,
                                                                               ham_freq_word, spam_freq_word, 1, s)

    accuracy, _, _, _, _ = test(ham_data, spam_data, ham_word_probabilities, spam_word_probabilities, ham_data, spam_data)
    print("Native Bayes, alpha = 1 - train:", 1 - accuracy)
    accuracy, _, _, _, _ = test(ham_test, spam_test, ham_word_probabilities, spam_word_probabilities, ham_test, spam_test)
    print("Native Bayes, alpha = 1 - test:", 1 - accuracy)

    ham_word_probabilities, spam_word_probabilities = cond_prob(ham_data_count, spam_data_count,
                                                                ham_freq_word, spam_freq_word, 0.1, s)
    _, true_positive, false_negative, false_positive, true_negative = test(ham_data, spam_data,
                                                                                    ham_word_probabilities,
                                                                                    spam_word_probabilities,
                                                                                    ham_test, spam_test)

    plot_confusion_matrix(true_positive, false_negative, false_positive, true_negative)

    false_positive_rate = false_positive / (false_positive + true_negative)
    false_negative_rate = false_negative / (false_negative + true_positive)

    print("False Positive Spam:", false_positive_rate)
    print("False Negative Spam:", false_negative_rate)

    k = 5
    alpha_values = [0.01, 0.1, 1]

    accuracies = k_fold_cross_validation(ham_data, spam_data, k, alpha_values)

    best_alpha_index = accuracies.index(max(accuracies))
    best_alpha = alpha_values[best_alpha_index]
    print("Optimalis alfa ertek 5-szoros kereszt-validalast alkalmazva:", best_alpha)

    ham_word_probabilities, spam_word_probabilities = semi_supervised_learning(ham_data, spam_data, 5,
                                                                               0.1, stopwords)

    accuracy, _, _, _, _ = test(ham_test, spam_test, ham_word_probabilities, spam_word_probabilities, ham_test, spam_test)
    print("Naive Bayes, felig felugyelt tanitassal - test:", 1 - accuracy)

if __name__ == "__main__":
    main()