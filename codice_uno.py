# Progetto Linguistica Computazionale 12 CFU
# Miriam Grande
# Programma 1

import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, FreqDist
from nltk.corpus import wordnet
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from nltk.corpus import movie_reviews

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("programma1.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


for resource in [
    "punkt",
    "punkt_tab",             
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",  
    "wordnet",
    "omw-1.4",
    "movie_reviews",
]:
    nltk.download(resource, quiet=True)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN 

def analyze_corpus(text):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    all_tokens = [token for sublist in tokens for token in sublist]
    pos_tags = pos_tag(all_tokens)
    tokens_no_punct_tags = [(tok, tag) for tok, tag in pos_tags if tok not in string.punctuation]
    lemmatizer = nltk.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(tok.lower(), get_wordnet_pos(tag)) for tok, tag in tokens_no_punct_tags]
    return {
        'sentences': sentences,
        'tokens': all_tokens,
        'pos_tags': pos_tags,
        'lemmas': lemmas
    }


def train_sentiment_classifier():
    documents = [(list(movie_reviews.words(fileid)), category) 
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    data = [" ".join(words) for words, label in documents]
    labels = [label for words, label in documents]

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(data)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=32)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return make_pipeline(vectorizer, model)

def classify_sentences(sentences, classifier):
    predictions = classifier.predict(sentences)
    pos_count = sum(1 for label in predictions if label == 'pos')
    neg_count = sum(1 for label in predictions if label == 'neg')
    document_score = pos_count - neg_count
    if document_score > 0:
        document_polarity = 'POS'
    elif document_score < 0:
        document_polarity = 'NEG'
    else:
        document_polarity = 'NEUTRA (pareggio)'

    return {
        'positive': pos_count,
        'negative': neg_count,
        'document_score': document_score,
        'document_polarity': document_polarity
    }

def analisi_finale(corpus1, corpus2, classifier):
    results = {}
    for name, corpus in [(' Corpus The Great Gatsby', corpus1), ('Corpus Articolo Scientifico', corpus2)]:
        analyzed = analyze_corpus(corpus)
        sentences = analyzed['sentences']
        tokens = analyzed['tokens']
        pos_tags = analyzed['pos_tags']
        lemmas = analyzed['lemmas']

        num_sentences = len(sentences)
        num_tokens = len(tokens)

        tokens_snz_punt = [t for t in tokens if t not in string.punctuation]
        avg_sentence_length = len(tokens_snz_punt) / num_sentences
        avg_token_length = sum(len(t) for t in tokens_snz_punt) / len(tokens_snz_punt)

        pos_dist = FreqDist(tag for _, tag in pos_tags[:1000])

        vocab = set(tokens_snz_punt)
        vocab_size = len(vocab)

        ttr = []
        for i in range(200, len(tokens_snz_punt) + 1, 200):
            ttr.append(len(set(tokens_snz_punt[:i])) / i)

        if len(tokens_snz_punt) % 200 != 0:
            ttr.append(len(set(tokens_snz_punt)) / len(tokens_snz_punt))

        lemma_vocab = set(lemmas)
        avg_lemmas_per_sentence = len(lemmas) / num_sentences

        sentence_polarity = classify_sentences(sentences, classifier)

        results[name] = {
            'Numero di frasi': num_sentences,
            'Numero di token': num_tokens,
            'Lunghezza media delle frasi': round(avg_sentence_length, 3),
            'Lunghezza media dei token': round(avg_token_length, 3),
            'Distribuzione delle POS nei primi 1000 token': pos_dist,
            'Dimensione del vocabolario': vocab_size,
            'TTR per porzioni incrementali di 200 token': [round(t, 3) for t in ttr],
            'Dimensione del vocabolario dei lemmi': len(lemma_vocab),
            'Numero medio di lemmi per frase': round(avg_lemmas_per_sentence, 3),
            'Numero di frasi classificate in POS': sentence_polarity['positive'],
            'Numero di frasi classificate in NEG': sentence_polarity['negative'],
            'Punteggio di polarità del documento (somma polarità frasi)': sentence_polarity['document_score'],
            'Polarità complessiva del documento': sentence_polarity['document_polarity']
        }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Confronta due corpora testuali (Programma 1 - Linguistica Computazionale)."
    )
    parser.add_argument(
        "corpus1", nargs="?", default="greatgatsby.txt",
        help="Path al primo file di testo, UTF-8 (default: greatgatsby.txt)"
    )
    parser.add_argument(
        "corpus2", nargs="?", default="articoloscientifico.txt",
        help="Path al secondo file di testo, UTF-8 (default: articoloscientifico.txt)"
    )
    parser.add_argument(
        "-o", "--output", default="Output_primocodice_BOH.txt",
        help="Nome del file di output (default: Output_primocodice.txt)"
    )
    args = parser.parse_args()

    logger.info("Caricamento dei corpora: '%s' e '%s'", args.corpus1, args.corpus2)
    try:
        with open(args.corpus1, "r", encoding="utf-8") as f:
            corpus1 = f.read()
        with open(args.corpus2, "r", encoding="utf-8") as f:
            corpus2 = f.read()
    except FileNotFoundError as e:
        logger.error("Impossibile trovare il file '%s'.", e.filename)
        logger.error("Controlla il path, oppure passa i file da riga di comando, es.:")
        logger.error("  python codice_uno.py percorso/corpus1.txt percorso/corpus2.txt")
        return

    logger.info("Addestramento del classificatore di sentiment (Naive Bayes su movie_reviews)...")
    classifier = train_sentiment_classifier()

    logger.info("Analisi linguistica dei due corpora in corso...")
    results = analisi_finale(corpus1, corpus2, classifier)

    logger.info("Scrittura dei risultati in '%s'...", args.output)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("Progetto di Linguistica Computazionale.\nProgramma Python n.1\n\n")

        for name, metrics in results.items():
            f.write(f"Analisi del {name}:\n")

            for metric, value in metrics.items():
                if metric == 'Distribuzione delle POS nei primi 1000 token':
                    f.write(f"\n{metric}:\n")

                    for pos, count in value.items():
                        f.write(f"{pos}: {count}\n")

                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("\n")

    logger.info("Fatto. Risultati scritti in '%s'.", args.output)


if __name__ == "__main__":
    main()
