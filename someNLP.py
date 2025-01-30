from nltk import ne_chunk, pos_tag, word_tokenize
from sumy.parsers.plaintext import PlaintextParser
from collections import defaultdict
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

"""
The goal of this script is to demonstrates additional NLP techniques I learned myself while taking the information retrieval course.
I will be working on:
1) Topic Modeling (using LDA or NMF)
2) Named Entity Recognition (using NLTK)
3) Text Summarization (using gensim)
"""



def perform_lda_topic_modeling(doc_term_matrix, vectorizer, n_topics=5, n_top_words=10):
    """
    Perform LDA topic modeling using a precomputed TF-IDF document-term matrix.

    Args:
        doc_term_matrix (sparse matrix): Precomputed TF-IDF document-term matrix.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used to transform text.
        n_topics (int): The number of topics.
        n_top_words (int): The number of top words per topic.

    Returns:
        dict: A dictionary where each topic ID maps to its top words.
    """
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()

    top_terms = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_feature_indices = topic.argsort()[::-1][:n_top_words]
        top_features = [feature_names[i] for i in top_feature_indices]
        top_terms[topic_idx] = top_features

    print("\n=== LDA Topic Modeling Results ===")
    for topic_id, words in top_terms.items():
        print(f"Topic {topic_id+1}: {', '.join(words)}")

    return top_terms


def perform_nmf_topic_modeling(doc_term_matrix, vectorizer, n_topics=5, n_top_words=10):
    """
    Perform NMF topic modeling using a precomputed TF-IDF document-term matrix.

    Args:
        doc_term_matrix (sparse matrix): Precomputed TF-IDF document-term matrix.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer used to transform text.
        n_topics (int): The number of topics.
        n_top_words (int): The number of top words per topic.

    Returns:
        dict: A dictionary where each topic ID maps to its top words.
    """
    nmf_model = NMF(n_components=n_topics, random_state=42)
    W = nmf_model.fit_transform(doc_term_matrix)
    H = nmf_model.components_

    feature_names = vectorizer.get_feature_names_out()

    top_terms = {}
    for topic_idx, topic in enumerate(H):
        top_feature_indices = topic.argsort()[::-1][:n_top_words]
        top_features = [feature_names[i] for i in top_feature_indices]
        top_terms[topic_idx] = top_features

    print("\n=== NMF Topic Modeling Results ===")
    for topic_id, words in top_terms.items():
        print(f"Topic {topic_id+1}: {', '.join(words)}")

    return top_terms




def extract_named_entities_nltk(doc_collection, max_docs=5):
    """
    Extract named entities from each document and summarize the entity distribution.

    Args:
        doc_collection (dict): Dictionary where each key is a document ID and its value is a document text.
        max_docs (int): Number of documents to process.

    Returns:
        dict: Dictionary mapping entity types to counts.
    """
    doc_ids = list(doc_collection.keys())[:max_docs]
    entity_distribution = defaultdict(int)

    for doc_id in doc_ids:
        text = doc_collection[doc_id]
        tokens = word_tokenize(text)  # Tokenization
        pos_tags = pos_tag(tokens)    # POS tagging
        chunks = ne_chunk(pos_tags, binary=False)  # NER

        for chunk in chunks:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_distribution[entity_type] += 1

    print("\nSummary of Named Entities Across Documents:")
    for entity_type, count in entity_distribution.items():
        print(f"{entity_type}: {count} occurrences")

    return entity_distribution



def summarize_documents(doc_collection, doc_term_matrix, vectorizer, max_docs=5, ratio=0.2):
    """
    Summarize documents using Sumy's LSA summarizer and print key insights.

    Args:
        doc_collection (dict): {doc_id: document text}.
        doc_term_matrix (sparse matrix): TF-IDF matrix from perform_clustering().
        vectorizer (TfidfVectorizer): Vectorizer instance used in TF-IDF transformation.
        max_docs (int): Number of documents to summarize.
        ratio (float): Approximate fraction of sentences to keep.

    Returns:
        dict: Dictionary mapping document IDs to summaries.
    """
    doc_ids = list(doc_collection.keys())[:max_docs]
    summaries = {}

    for i, doc_id in enumerate(doc_ids):
        text = doc_collection[doc_id]
        parser = PlaintextParser.from_string(text, Tokenizer("english"))

        # Select key terms for this document
        term_weights = doc_term_matrix[i].toarray().flatten()
        important_words = [vectorizer.get_feature_names_out()[j] for j in term_weights.argsort()[-5:]]  # Top 5 words

        # Extract sentences containing key terms
        selected_sentences = [str(sent) for sent in parser.document.sentences if any(term in str(sent).lower() for term in important_words)]
        num_sentences = max(1, int(len(parser.document.sentences) * ratio))

        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, num_sentences)
        summary_text = ' '.join(str(sentence) for sentence in summary_sentences)

        summaries[doc_id] = summary_text

    print("\nSummary of Extracted Key Sentences:")
    for doc_id, summary in summaries.items():
        print(f"\nDocument ID: {doc_id}")
        print("Summary:\n", summary[:300], "...")  # Print first 300 characters

    return summaries

