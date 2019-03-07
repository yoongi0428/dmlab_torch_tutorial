import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams

def calculate_bleu(pred, gold):

    score = corpus_bleu(gold, pred) * 100

    return score



def vis_attention(s1, s2, attention):
    pass