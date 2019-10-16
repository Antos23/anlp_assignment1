import re
import random
import math
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


# task1
def preprocess_line(str):
    # remove the other characters
    new_str = re.sub('[^a-zA-Z0-9. ]', '', str)
    # convert all digits to 0
    new_str = re.sub('[0-9]', "0", new_str)
    # convert all English characters to lower case
    new_str = new_str.lower()
    # add '##' at the beginning and '#' at the end of each line
    new_str = '##' + new_str + '#'
    # avoid double spaces
    new_str = ' '.join(new_str.split())

    return new_str


# Task 3
vocab = [' ', '#', '.', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
         's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


# count n-gram sequence in each line
def count_ngrams(ngram_count, str, n):
    for i in range(0, len(str) - n + 1):
        ngram = str[i:i + n]
        if ngram not in ngram_count:
            ngram_count[ngram] = 1
        else:
            ngram_count[ngram] += 1
    return ngram_count


# count bigrams and trigrams of a text and generates trigram probabilities using add-alpha smoothing
def language_model(input_file, language, alpha):
    bigram_count = {}
    trigram_count = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        bigram_count = count_ngrams(bigram_count, line, 2) 
        trigram_count = count_ngrams(trigram_count, line, 3)

    # estimate trigram probabilities
    prob = {}
    for c1 in vocab:
        for c2 in vocab:
            for c3 in vocab:
                # to avoid sequences like '# #' and ' # '
                if c1 == '#' and c3 == '#':
                    continue
                if c1 != '#' and c2 == '#':
                    continue
                seq2 = ''.join([c1, c2])
                seq3 = ''.join([c1, c2, c3])
                if seq2 not in bigram_count:
                    bigram_count[seq2] = 0
                if seq3 not in trigram_count:
                    trigram_count[seq3] = 0

                # add alpha smoothing
                if c1 == '#' and c2 != '#':
                    prob[seq3] = (trigram_count[seq3] + alpha) / (bigram_count[seq2] + alpha * 29)
                else:
                    prob[seq3] = (trigram_count[seq3] + alpha) / (bigram_count[seq2] + alpha * 30)

    # write the trigram model probabilities into file
    output_file = open('trigram_model.' + language, 'w')
    for item in prob:
        # output_file.write(item + '\t' + str(prob[item]) + '\n')
        output_file.write(item + '\t' + '%e' % prob[item] + '\n')

    output_file.close()


# Split test text into 2 parts: a held-out (validation) text and a test text

def split_input_file(input_file):
    text = []
    with open(input_file) as f:
        for line in f:
            line = preprocess_line(line)
            text.append(line)
        validation, training = train_test_split(text, train_size=0.2, random_state=1)

    # save validation text into txt file
    with open("validation", "w") as f:
        for line in validation:
            f.write("".join(line) + "\n")
            # save test text into txt file
    with open("new_train", "w") as f:
        for line in training:
            f.write("".join(line) + "\n")


# Train the training text with different alphas and choose the one that minimizes the perplexity on the validation test
def choose_alpha(train_file, validation_file, language):
    perplexities = dict()
    #we are iterating over such a small value of alpha, with a very small step as we have already iterated from 0.1 to 1 with a 0.1 step.
    #this is to narrow down the value of alpha without causing a long running time
    for alpha in np.arange(0.01, 0.3, 0.01): 
        training = open(train_file, 'r')
        language_model(training, language, alpha)  # generate a model for each value of alpha in the range
        perplexities[alpha] = calculate_perplexity('trigram_model.'+language,
                                                   validation_file)  # compute perplexity on the validation text
    # Save alpha that minimizes perplexity
    best_alpha = min(perplexities, key=perplexities.get)
    # Plot perplexities
    plt.plot(list(perplexities.keys()), list(perplexities.values()))
    plt.plot(best_alpha, perplexities[best_alpha], marker='o')
    plt.xlabel("alpha")
    plt.ylabel("Perplexity")
    plt.show()
    return best_alpha

# Task 4
N = 300

def generate_from_LM(model_file_name):
    f = open(model_file_name)
    # read the estimated probability
    model = {}
    for line in f:
        line = line.split('\t')
        model[line[0]] = float(line[1])

    output = ''
    head = '##'
    output += head
    # generate the other characters
    while len(output) < N:
        population = [k for (k, v) in model.items() if k.startswith(head)]
        weights = [model[k] for k in population]
        trigram_picked, = random.choices(population=population, weights=weights, k=1)
        # print(trigram_picked)
        ch_picked = trigram_picked[-1]
        output += ch_picked
        head = output[-2:]
        if ch_picked == '#':
            head = '##'
            output += '\n##'

    return (output)

# Task 5

def calculate_perplexity(model, test_file):
    # read model
    f1 = open(model, 'r')
    prob = {}
    for line in f1:
        line = line.split('\t')
        prob[line[0]] = float(line[1])

    total_logp = 0
    count = 0

    # read test file
    f2 = open(test_file, 'r', encoding="ISO-8859-1")
    for line in f2:
        # process line as described in task1
        line = preprocess_line(line)
        for i in range(0, len(line) - 2):
            p = prob[line[i:i + 3]]
            total_logp += -math.log2(p)
            count += 1

    Hm = total_logp / count
    PPm = 2 ** Hm

    f1.close()
    f2.close()
    return (PPm)

if __name__ == '__main__':
    # task3
    split_input_file('./data/training.en')
    input_file = open('./data/training.en', 'r')
    language_model(input_file, 'en', choose_alpha('new_train', 'validation', 'en'))
    input_file.close()

    split_input_file('./data/training.es')
    input_file = open('./data/training.es', 'r')
    language_model(input_file, 'es', choose_alpha('new_train', 'validation', 'es'))
    input_file.close()

    split_input_file('./data/training.de')
    input_file = open('./data/training.de', 'r')
    language_model(input_file, 'de', choose_alpha('new_train', 'validation', 'de'))
    input_file.close()
    # input_file.close()

    # task4
    print('output of model-br.en:')
    print(generate_from_LM('./data/model-br.en'))
    print('output of our English language model:')
    print(generate_from_LM('trigram_model.en'))

    # task5
    print('perplexity on English model:')
    print(calculate_perplexity('trigram_model.en', './data/test'))
    print('perplexity on Spanish model:')
    print(calculate_perplexity('trigram_model.es', './data/test'))
    print('perplexity on German model:')
    print(calculate_perplexity('trigram_model.de', './data/test'))

