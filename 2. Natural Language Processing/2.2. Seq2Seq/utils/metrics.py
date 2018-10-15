from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.util import ngrams

def bleu(src, tar):
    return 1.0