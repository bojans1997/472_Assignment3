import csv

class NaiveBayes:
    training_set = []
    smoothing = 0.01
    filtered = False

    numFactualWords = 0
    numNonFactualWords = 0
    numFactualTweets = 0
    numNonFactualTweets = 0
    vocabularySize = 0

    probFactual = 0
    probNonFactual = 0

    wordClassCount = {}
    conditionals = {}

    def __init__(self, training_set, filtered = False):
        self.training_set = training_set
        self.filtered = filtered

        self.getWordFrequencies()
        self.computeClassProbs()
        self.computeConditionals()

    def getWordFrequencies(self):
    # get word frequency for each class (yes / no)
        for entry in self.training_set:
            entry_text = entry[1].split()
            for word in entry_text:
                word = word.lower()
                word = word.strip(".,:\"“-—'")
                if (word, "factual") in self.wordClassCount.keys() or (word, "nonFactual") in self.wordClassCount.keys():
                    if entry[2] == "yes":
                        self.numFactualWords += 1
                        self.wordClassCount[(word, "factual")] += 1
                    else:
                        self.numNonFactualWords += 1
                        self.wordClassCount[(word, "nonFactual")] += 1
                else:
                    if entry[2] == "yes":
                        self.numFactualWords += 1
                        self.wordClassCount[(word, "factual")] = 1
                        self.wordClassCount[(word, "nonFactual")] = 0
                    else:
                        self.numNonFactualWords += 1
                        self.wordClassCount[(word, "factual")] = 0
                        self.wordClassCount[(word, "nonFactual")] = 1

        # filter out words that only appear once if filtered is set to True
        words_to_remove = []
        if self.filtered:
            for word in self.wordClassCount.keys():
                if self.wordClassCount[(word[0], "factual")] + self.wordClassCount[(word[0], "nonFactual")] < 2:
                    words_to_remove.append(word)
            for word in words_to_remove:
                if word[1] == "factual":
                    self.numFactualWords -= 1
                else:
                    self.numNonFactualWords -= 1
                del self.wordClassCount[word]
        self.vocabularySize = len(self.wordClassCount) / 2
    
    def computeClassProbs(self):
    # get probability of each class (yes / no)
        for entry in self.training_set:
            if entry[2] == "yes":
                self.numFactualTweets += 1
            else:
                self.numNonFactualTweets += 1
        self.probFactual = self.numFactualTweets / len(self.training_set)
        self.probNonFactual = self.numNonFactualTweets / len(self.training_set)

    def computeConditionals(self):
    # get value of conditionals for each word
        for word in self.wordClassCount.keys():
            if word[1] == "factual":
                self.conditionals[word] = self.wordClassCount[word] / self.numFactualWords
            else:
                self.conditionals[word] = self.wordClassCount[word] / self.numNonFactualWords

training_data = list(csv.reader(open('Resources/covid_training.tsv', encoding="utf-8"), delimiter='\t'))[1:]

nb = NaiveBayes(training_data, True)
#nb.train()