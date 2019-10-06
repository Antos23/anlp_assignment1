import re
from itertools import permutations

# task1
def preprocess_line(str):
    # remove the other characters
    fil1 = re.compile('[^a-zA-Z0-9. ]')
    new_str = fil1.sub('', str)

    # convert all digits to 0
    #fil2 = re.compile('[0-9]')
    #new_str = fil2.sub('0', new_str)
    new_str = re.sub('[0-9]', "0", new_str) #to compress the two lines above 

    # convert all English characters to lower case
    new_str = new_str.lower()
    
    #to add # at the beginning and at end of each line
    new_str = "##" + new_str + "#"

    return new_str


# task2
# Q: By looking at the language model probabilities in this file, can you say anything about the kind of estimation
# method that was used?
# A: Maximum likelihood estimation with add one smoothing. Because all unseen 3-character sequences have the same
# probability, and not small enough. Just as 'steal' too much from the other right sequences. So it's using add one smoothing.


# task3
# IMPORTANT QUESTION: DO WE NEED TO ADD '#' AT THE BEGINNING AND THE END OF THE SENTENCE???
# We are adding "##" at the beginning and "#" at the end of each line
#alpha = [' ', '.', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
         't', 'u', 'v', 'w', 'x', 'y', 'z']
alpha = [' ', '#', '.', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
          's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# count n-characters sequence in each line
def count_character(count_result, str, n):
    for i in range(0, len(str) - n + 1):
        seq = str[i:i + n]
        if seq not in count_result:
            count_result[seq] = 1
        else:
            count_result[seq] += 1
    return count_result


def language_model(input_file, language):
    count_character_2 = {}
    count_character_3 = {}
    for line in input_file:
        # process line as described in task1
        line = preprocess_line(line)
        count_character_2 = count_character(count_character_2, line, 2)
        count_character_3 = count_character(count_character_3, line, 3)

    # estimate trigram probabilities
    prob = {}
    #for c1 in alpha:
        #for c2 in alpha:
            #for c3 in alpha:
                #seq2 = ''.join([c1, c2])
                #seq3 = ''.join([c1, c2, c3])
     # an alternative possible way to obtain all the permutations is using the itertools package
     seq2 = [''.join(p) for p in permutations(alpha, 2)]
     seq3 = [''.join(p) for p in permutations(alpha, 3)]
     if seq2 not in count_character_2:
        count_character_2[seq2] = 0
     if seq3 not in count_character_3:
        count_character_3[seq3] = 0

      # add one smoothing
      prob[seq3] = (count_character_3[seq3] + 1) / (count_character_2[seq2] + 29) #or +len(alpha)?
             

    # write the trigram model probabilities into file
    output_file = open('trigram_model.' + language, 'w')
    for item in prob:
        output_file.write(item + '\t' + str(prob[item]) + '\n')

    output_file.close()


# task4
import random

N = 300


# roulette algorithm to choose the higher probability character. assume the sum of probability is 1
# if don't use this algorithm, the program will always choose character ' ', 't', 'h', 'e' and output the the the.....
def choose_character(new_prob):
    keys = list(new_prob.keys())
    values = list(new_prob.values())
    target = random.uniform(0, 1)
    print(target)
    present = 0
    k = ''
    for i in range(0, len(keys)):
        present += values[i]
        if present >= target:
            k = keys[i]
            break

    return k


def generate_from_LM(model_file_name):
    f = open(model_file_name)
    # read the estimated probability
    prob = {}
    for line in f:
        line = line.split('\t')
        prob[line[0]] = float(line[1])

    output_str = ''

    # random generate the first two characters
    head = random.sample(alpha, 2)
    head = ''.join(head)
    output_str += head

    # generate the other characters
    for i in range(N - 2):
        print(head)
        new_prob = {}
        for item in alpha:
            new_prob[item] = prob[head + item]

        print(new_prob)
        k = choose_character(new_prob)
        # k = max(new_prob.items(), key=lambda x: x[1])
        # output_str += k[0]
        output_str += k
        head = output_str[-2:]

    print(output_str)


# task5
import math


# the slide only give the function for one sequence. so i don't know whether this function is right for all lines
# in the test file
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
    f2 = open(test_file, 'r')
    for line in f2:
        # process line as described in task1
        line = process_line(line)
        for i in range(0, len(line) - 2):
            p = prob[line[i:i + 3]]
            total_logp += -math.log2(p)
            count += 1

    Hm = total_logp / count
    PPm = 2 ** Hm
    print(PPm)


# task6


if __name__ == '__main__':
    # task3
    # input_file = open('./data/training.en', 'r')
    # language_model(input_file, 'en')
    # input_file = open('./data/training.es', 'r')
    # language_model(input_file, 'es')
    # input_file = open('./data/training.de', 'r')
    # language_model(input_file, 'de')
    # input_file.close()

    # task4
    # generate_from_LM('./data/model-br.en')
    # generate_from_LM('trigram_model.en')

    # task5
    calculate_perplexity('trigram_model.en', './data/test')
    calculate_perplexity('trigram_model.es', './data/test')
    calculate_perplexity('trigram_model.de', './data/test')
