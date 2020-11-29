import csv
import math

class NaiveBayes:
    def __init__(self, training_set, test_set, filtered = False):
        self.training_set = training_set
        self.test_set = test_set
        self.filtered = filtered
        self.smoothing = 0.01
        self.traceFileName = "trace_NB-BOW-OV.txt"
        self.evalFileName = "eval_NB-BOW-OV.txt"

        self.numFactualWords = 0
        self.numNonFactualWords = 0
        self.numFactualTweets = 0
        self.numNonFactualTweets = 0
        self.vocabulary = set()
        self.vocabularySize = 0

        self.probFactual = 0
        self.probNonFactual = 0

        self.wordClassCount = {}
        self.conditionals = {}
        if self.filtered:
            self.traceFileName = "trace_NB-BOW-FV.txt"
            self.evalFileName = "eval_NB-BOW-FV.txt"

        self.getWordFrequencies()
        self.computeClassProbs()
        self.computeConditionals()
        self.predict()

    def getWordFrequencies(self):
    # get word frequency for each class (yes / no)
        for entry in self.training_set:
            entry_text = entry[1].split()
            for word in entry_text:
                word = word.strip(".,?!@#:\"“-—'()").lower()
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
        for key in self.wordClassCount.keys():
            self.vocabulary.add(key[0])
    
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
                self.conditionals[word] = (self.wordClassCount[word] + self.smoothing) / (self.numFactualWords + (self.vocabularySize * self.smoothing))
            else:
                self.conditionals[word] = (self.wordClassCount[word] + self.smoothing) / (self.numNonFactualWords + (self.vocabularySize * self.smoothing))

    def predict(self):
        incorrectPredictions = 0
        predictedFactualCount = 0
        predictedNonFactualCount = 0
        incorrectFactualPredictions = 0
        incorrectNonFactualPredictions = 0
        actualFactualCount = 0
        actualNonFactualCount = 0

        f = open(self.traceFileName, "w")
        for entry in self.test_set:
            factualScore = math.log10(self.probFactual)
            nonFactualScore = math.log10(self.probNonFactual)
            for word in entry[1].split():
                word = word.strip(".,?!@#:\"“-—'()").lower()
                if word not in self.vocabulary:
                    continue
                factualScore = factualScore + math.log10(self.conditionals[word, "factual"])
                nonFactualScore = nonFactualScore + math.log10(self.conditionals[word, "nonFactual"])
            
            likely_class = "yes"
            score = '{:.2e}'.format(factualScore)
            if nonFactualScore > factualScore:
                likely_class = "no"
                score = '{:.2e}'.format(nonFactualScore)
            tweet_id = entry[0]
            correct_class = entry[2]
            label = "correct"
            if likely_class != correct_class:
                label = "wrong"
                incorrectPredictions += 1
                # for calculating precision
                if likely_class == "yes":
                    incorrectFactualPredictions += 1
                else:
                    incorrectNonFactualPredictions += 1
            
            # for calculating precision
            if likely_class == "yes":
                predictedFactualCount += 1
            else:
                predictedNonFactualCount += 1

            # for calculating recall
            if correct_class == "yes":
                actualFactualCount += 1
            else:
                actualNonFactualCount += 1

            f.write(tweet_id + "  " + likely_class + "  " + str(score) + "  " + correct_class + "  " + label + "\n")
        f.close()
        
        accuracy = (len(self.test_set) - incorrectPredictions) / len(self.test_set)
        yesPrecision = (predictedFactualCount - incorrectFactualPredictions) / predictedFactualCount
        noPrecision = (predictedNonFactualCount - incorrectNonFactualPredictions) / predictedNonFactualCount
        yesRecall = (actualFactualCount - incorrectNonFactualPredictions) / actualFactualCount
        noRecall = (actualNonFactualCount - incorrectFactualPredictions) / actualNonFactualCount
        yesF1 = (2 * yesPrecision * yesRecall) / (yesPrecision + yesRecall)
        noF1 = (2 * noPrecision * noRecall) / (noPrecision + noRecall)

        f = open(self.evalFileName, "w")
        f.write(str(accuracy) + "\n")
        f.write(str(yesPrecision) + "  " + str(noPrecision) + "\n")
        f.write(str(yesRecall) + "  " + str(noRecall) + "\n")
        f.write(str(yesF1) + "  " + str(noF1))
        f.close()

training_data = list(csv.reader(open('Resources/covid_training.tsv', encoding="utf-8"), delimiter='\t'))[1:]
test_data = list(csv.reader(open('Resources/covid_test_public.tsv', encoding="utf-8"), delimiter='\t'))

filteredNB = NaiveBayes(training_data, test_data, True)
unfilteredNB = NaiveBayes(training_data, test_data)