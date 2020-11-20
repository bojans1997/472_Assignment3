import csv
import re

class NaiveBayes:
    training_set = []
    smoothing = 0.01
    filtered = False
    trained_model = {}

    def __init__(self, training_set, filtered = False):
        self.training_set = training_set
        self.filtered = filtered

    # get number of times each word appears
    def train(self):
        for entry in self.training_set:
            entry_text = entry[1].split()
            for word in entry_text:
                word = word.lower()
                word = word.rstrip(".")
                word = word.rstrip(",")
                word = word.rstrip(":")
                word = word.lstrip(".")
                word = word.lstrip(",")
                word = word.lstrip(":")
                if word in self.trained_model.keys():
                    self.trained_model[word] += 1
                else:
                    self.trained_model[word] = 1

        # filter out words that only appear once
        words_to_remove = []
        if self.filtered:
            for word in self.trained_model:
                if self.trained_model[word] < 2:
                    words_to_remove.append(word)
            for word in words_to_remove:
                del self.trained_model[word]

        for x in self.trained_model:
            print(x + " " + str(self.trained_model[x]))
    
    def predict(self):
        return False


training_data = list(csv.reader(open('Resources/covid_training.tsv', encoding="utf-8"), delimiter='\t'))[1:]

#for entry in training_data:
#    print(entry)

nb = NaiveBayes(training_data)
nb.train()