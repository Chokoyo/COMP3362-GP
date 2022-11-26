import os
import editdistance
from nltk.translate.bleu_score import sentence_bleu

class MetricInterface:
    def __init__(self, target, pred):
        """
        Constructor 
        Input:
            target: a list of target tokens of type numpy.str_
            pred: a list of predicted tokens of type numpy.str_
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the metric to its initial state.
        """
        raise NotImplementedError

    def result(self):
        """
        Return the Score of the metric.
        """
        raise NotImplementedError
    

class Metric(MetricInterface):
    def __init__(self, target, pred):
        self.target = target
        self.pred = pred
        self._result = None

    def reset(self):
        self._result = None

    def result(self, metric="edit_distance"):
        if metric == "edit_distance":
            self._result = self.edit_distance_score()
        elif metric == "bleu":
            self._result = self.corpus_bleu_score()
        return self._result

    def edit_distance_score(self, has_space=False):
        """
        Calculate the edit distance score based on Levenstein's algorithm.
        """
        total_edit_distance = 0
        total_ref = 0
        for t, p in zip(self.target, self.pred):
            t_tokens = t.split()
            p_tokens = p.split()
            if has_space:
                pass
            t_tokens = tuple(t_tokens)
            p_tokens = tuple(p_tokens)
            total_edit_distance += editdistance.eval(t_tokens, p_tokens)
            total_ref += max(len(t_tokens), len(p_tokens))
        return 1. - float(total_edit_distance / total_ref)

    def corpus_bleu_score(self, weights=(0.25, 0.25, 0.25, 0.25)):
        """
        Calculate the BLEU score with 4-gram on the entire corpus.
        to check the score of each unigram, set weights=(1, 0, 0, 0)
        bigram: weights=(0, 1, 0, 0), etc.

        """
        reference = [[]]
        candidate = []
        for t, p in zip(self.target, self.pred):
            t_tokens = t.split()
            p_tokens = p.split()
            reference[0] += t_tokens
            candidate += p_tokens
        return sentence_bleu(reference, candidate, weights=weights)
