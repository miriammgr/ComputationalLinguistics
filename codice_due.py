# Progetto Linguistica Computazionale 12 CFU
# Miriam Grande, matricola 665939
# Programma 2

import logging
import nltk
from collections import Counter
from nltk import ngrams, pos_tag, word_tokenize, ne_chunk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from math import log

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("programma2.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

for resource in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
    "maxent_ne_chunker",
    "maxent_ne_chunker_tab",
    "words",
]:
    nltk.download(resource, quiet=True)

def extract_pos(tokens, pos_list):
    tagged_tokens = pos_tag(tokens)
    return [word for word, pos in tagged_tokens if pos in pos_list]


def extract_bigrams(tokens, pos_tags_1, pos_tags_2):
    tagged_tokens = pos_tag(tokens)
    bigrams = [((w1, t1), (w2, t2)) for (w1, t1), (w2, t2) in ngrams(tagged_tokens, 2)]
    return [(w1, w2) for (w1, t1), (w2, t2) in bigrams if t1 in pos_tags_1 and t2 in pos_tags_2]


def get_top_ngrams(tokens, n, top_k):
    n_grams = ngrams(tokens, n)
    n_gram_freq = Counter(n_grams)
    return n_gram_freq.most_common(top_k)


def get_top_pos_ngrams(tagged_tokens, n, top_k):
    pos_ngrams = ngrams(tagged_tokens, n)
    pos_ngram_freq = Counter(pos_ngrams)
    return pos_ngram_freq.most_common(top_k)

def calculate_bigram_metrics(bigrams, tokens):
    bigram_freq = Counter(bigrams)
    unigram_freq = Counter(tokens)
    total_tokens = len(tokens)

    metrics = []
    for bigram, freq in bigram_freq.items():
        w1, w2 = bigram
        p_w1 = unigram_freq[w1] / total_tokens if total_tokens > 0 else 0

        p_bigram = freq / total_tokens if total_tokens > 0 else 0
        conditional_prob = p_bigram / p_w1 if p_w1 > 0 else 0
        joint_prob = conditional_prob * p_w1 if conditional_prob > 0 else 0
        p_w2 = unigram_freq[w2] / total_tokens if total_tokens > 0 else 0
        mi = log(p_bigram / (p_w1 * p_w2), 2) if p_w1 > 0 and p_w2 > 0 and p_bigram > 0 else 0
        lmi = freq * mi
        metrics.append((bigram, freq, conditional_prob, joint_prob, mi, lmi))

    return metrics

def calculate_stopwords_perc(tokens):
    stop_words = set(stopwords.words('english'))
    stopwords_count = sum(1 for token in tokens if token.lower() in stop_words)
    total_tokens = len(tokens)
    percentage = (stopwords_count / total_tokens) * 100 if total_tokens > 0 else 0
    return percentage

def analyze_sentence(sentences, unigram_freq, bigram_freq, trigram_freq, alpha=1):
    results = []
    for i, sentence in enumerate(sentences):
        tokens = [t.lower() for t in word_tokenize(sentence) if t.isalpha()]

        if 10 <= len(tokens) <= 20:
            frequent_tokens = sum(1 for token in tokens if unigram_freq[token] >= 2)
            if frequent_tokens >= len(tokens) // 2:
                freq_sum = sum(unigram_freq[token] for token in tokens)
                avg_freq = freq_sum / len(tokens)
                trigrams = list(ngrams(tokens, 3))
                log_prob_markov = 0
                for trigram in trigrams:
                    w1, w2, w3 = trigram
                    context_freq = bigram_freq.get((w1, w2), 0)
                    prob_trigram = (trigram_freq.get((w1, w2, w3), 0) + alpha) / (context_freq + alpha * len(unigram_freq))
                    if prob_trigram > 0:
                        log_prob_markov += log(prob_trigram)
                    else:
                        log_prob_markov += float('-inf') 

                results.append((sentence, avg_freq, log_prob_markov))

    if results:
        highest_avg = max(results, key=lambda x: x[1])
        lowest_avg = min(results, key=lambda x: x[1])
        highest_prob = max(results, key=lambda x: x[2])

        return results, highest_avg, lowest_avg, highest_prob
    else:
        return results, None, None, None


def calculate_pronoun_freq(sentences):
    total_tokens = 0
    total_pronouns = 0
    total_pronouns_per_sentence = 0

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        total_tokens += len(tokens)
        pronouns_in_sentence = sum(1 for token, pos in pos_tags if pos == 'PRP')
        total_pronouns += pronouns_in_sentence
        total_pronouns_per_sentence += pronouns_in_sentence

    pronoun_frequency = total_pronouns / total_tokens if total_tokens > 0 else 0

    average_pronouns_per_sentence = total_pronouns_per_sentence / len(sentences) if len(sentences) > 0 else 0

    return pronoun_frequency, average_pronouns_per_sentence


def extract_ne(tokens):

    tagged_tokens = pos_tag(tokens)
    chunked = ne_chunk(tagged_tokens)

    named_entities = []
    for tree in chunked:
        if isinstance(tree, nltk.Tree): 
            entity = " ".join([word for word, tag in tree])
            entity_type = tree.label()
            named_entities.append((entity, entity_type))

    return named_entities

def extract_and_count_ne(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens_no_stop = [token for token in tokens if token.isalpha() and token.lower() not in stop_words]
    named_entities = extract_ne(tokens_no_stop)

    entity_counter = {}
    for entity, entity_type in named_entities:
        entity_counter.setdefault(entity_type, Counter())[entity] += 1

    return entity_counter

def find_common_elements(metrics):
    top_10_mi = set([x[0] for x in sorted(metrics, key=lambda x: x[4], reverse=True)[:10]])
    top_10_lmi = set([x[0] for x in sorted(metrics, key=lambda x: x[5], reverse=True)[:10]])
    common_elements = top_10_mi.intersection(top_10_lmi)
    return len(common_elements), common_elements

def analisi_finale_stampa(input_file, output_file):
    with open(output_file, 'w', encoding='utf-8') as f, open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

        f.write("Progetto di Linguistica Computazionale.\nProgramma Python n.2\n\n")

        stampa_finale = []
        tokens = word_tokenize(text)
        logger.info("Corpus '%s': %d token totali. Tokenizzazione completata.", input_file, len(tokens))
        stop_words = set(stopwords.words('english'))
        tokens_no_stop = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stop_words]

        logger.info("Calcolo PoS tagging e top-50 sostantivi/avverbi/aggettivi...")
        pos_to_extract = {
            'Sostantivi': ['NN', 'NNS', 'NNP', 'NNPS'],
            'Avverbi': ['RB', 'RBR', 'RBS'],
            'Aggettivi': ['JJ', 'JJR', 'JJS']
        }
        for pos_name, pos_tags in pos_to_extract.items():
            tokens_per_pos_cat = extract_pos(tokens_no_stop, pos_tags)
            freq_dist = FreqDist(tokens_per_pos_cat)
            stampa_finale.append(f"Top 50 {pos_name}:")
            for word, freq in freq_dist.most_common(50):
                stampa_finale.append(f"\t{word}: {freq}")
            stampa_finale.append("")

        for n in [1, 2, 3]:
            stampa_finale.append(f"Top 20 {n}-grammi più frequenti:")
            top_ngrams = get_top_ngrams(tokens_no_stop, n, 20)
            for ngram, freq in top_ngrams:
                stampa_finale.append(f"\t{' '.join(ngram)}: {freq}")
            stampa_finale.append("")

        tagged_tokens = [pos for _, pos in pos_tag(tokens_no_stop)]
        for n in range(1, 6):
            stampa_finale.append(f"Top 20 {n}-grammi di PoS più frequenti:")
            top_pos_ngrams = get_top_pos_ngrams(tagged_tokens, n, 20)
            for ngram, freq in top_pos_ngrams:
                stampa_finale.append(f"\t{' '.join(ngram)}: {freq}")
            stampa_finale.append("")

        stopwords_percentage = calculate_stopwords_perc(tokens)
        stampa_finale.append(f"Percentuale di stopwords nel corpus: {stopwords_percentage:.2f}%")

        logger.info("Calcolo delle metriche sui bigrammi verbo+sostantivo (freq, prob, MI, LMI)...")
        bigrams = extract_bigrams(tokens_no_stop, ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
                                ['NN', 'NNS', 'NNP', 'NNPS'])
        metrics = calculate_bigram_metrics(bigrams, tokens_no_stop)

        common_count, common_elements = find_common_elements(metrics)

        stampa_finale.append("\nTop-10 metriche dei bigrammi:")

        metric_names = ["Frequenza", "Probabilità condizionata", "Probabilità congiunta", "MI", "LMI"]
        for i, metric_name in enumerate(metric_names):
            metrics_sorted = sorted(metrics, key=lambda x: x[i + 1], reverse=True)[:10]
            stampa_finale.append(f"\nIn base a {metric_name}:")
            for metric in metrics_sorted:
                stampa_finale.append(f"{metric[0]} - {metric_name}: {round(metric[i + 1], 4)}")

        stampa_finale.append(f"\nNumero di elementi comuni ai top-10 per MI e LMI: {common_count}")
        stampa_finale.append("Elementi comuni:")
        if not common_elements:
            stampa_finale.append("\tNessun elemento in comune è stato trovato.")
        else:
            for element in common_elements:
                stampa_finale.append(str(element))

        logger.info("Analisi delle frasi (modello di Markov di ordine 2)...")
        sentences = nltk.sent_tokenize(text)
        tokens_all_lower = [t.lower() for t in tokens if t.isalpha()]
        unigram_freq_full = Counter(tokens_all_lower)
        bigram_freq_full = Counter(ngrams(tokens_all_lower, 2))
        trigram_freq_full = Counter(ngrams(tokens_all_lower, 3))
        results, highest_avg, lowest_avg, highest_prob = analyze_sentence(sentences, unigram_freq_full, bigram_freq_full, trigram_freq_full, alpha=1)

        if highest_avg:
            stampa_finale.append("\nAnalisi delle frasi:")
            stampa_finale.append("\na. Frase con la media di distribuzione di frequenza dei token più alta:")
            stampa_finale.append(f"\tFrase: {highest_avg[0]}")
            stampa_finale.append(f"\tFrequenza media: {highest_avg[1]:.4f}")

            stampa_finale.append("\nb. Frase con la media di distribuzione di frequenza dei token più bassa:")
            stampa_finale.append(f"\tFrase: {lowest_avg[0]}")
            stampa_finale.append(f"\tFrequenza media: {lowest_avg[1]:.4f}")

            stampa_finale.append("\nc. Frase con probabilità più alta secondo un modello di Markov di ordine II:")
            stampa_finale.append(f"\tFrase: {highest_prob[0]}")
            stampa_finale.append(f"\tLog probabilità: {highest_prob[2]:.4f}")
        else:
            stampa_finale.append("\n\tPer quanto riguarda l'analisi delle frasi, in questo corpus nessuna frase soddisfa i criteri.")  # (In uno dei due corpora nessuna frase soddisfa i criteri)

        pronoun_frequency, avg_pronouns_per_sentence = calculate_pronoun_freq(sentences)
        stampa_finale.append(f"\nNumero di pronomi personali sul totale di token: {pronoun_frequency:.4f}")
        stampa_finale.append(f"Numero medio di pronomi personali per frase: {avg_pronouns_per_sentence:.2f}")

        logger.info("Estrazione delle Entità Nominate (ne_chunk)...")
        entity_counter = extract_and_count_ne(text)
        logger.info("Trovate %d classi di NE: %s", len(entity_counter), list(entity_counter.keys()))
        for entity_type, counter in entity_counter.items():
            if counter:
                stampa_finale.append(f"\nTop 15 {entity_type} entities:")
                for entity, freq in counter.most_common(15):
                    stampa_finale.append(f"\t{entity}: {freq}")
            else:
                stampa_finale.append(f"\n\tNessuna NE trovata per la categoria {entity_type}")

        f.write('\n'.join(stampa_finale))

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Analizza uno o più corpora testuali (Programma 2 - Linguistica Computazionale)."
    )
    parser.add_argument(
        "corpora", nargs="*", default=["greatgatsby.txt", "articoloscientifico.txt"],
        help="Uno o più path a file di testo UTF-8 da analizzare "
             "(default: greatgatsby.txt articoloscientifico.txt)"
    )
    args = parser.parse_args()

    for corpus_path in args.corpora:
        base_name = os.path.splitext(os.path.basename(corpus_path))[0]
        output_path = f"Output_secondocodice_{base_name}.txt"
        logger.info("Elaborazione di '%s'...", corpus_path)
        try:
            analisi_finale_stampa(corpus_path, output_path)
        except FileNotFoundError:
            logger.error("Impossibile trovare il file '%s'. Salto questo corpus.", corpus_path)
            continue
        logger.info("Fatto. Risultati scritti in '%s'.", output_path)


if __name__ == "__main__":
    main()