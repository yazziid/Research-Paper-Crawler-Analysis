import numpy as np
import requests
from sklearn.metrics import silhouette_samples
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import urllib.robotparser
from io import BytesIO
import nltk
from collections import defaultdict
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import PyPDF2
from someNLP import *


def get_thesis_links(base_url, max_thesis):
    """
    Crawls the Spectrum library website to fetch thesis PDF links.

    Args:
        base_url (str): The base URL of the Spectrum library website.
        max_thesis (int): The maximum number of theses to fetch.

    Returns:
        list: A list of thesis URLs.
    """
    thesis_links = []
    visited_urls = set()
    urls_to_visit = [base_url]
    domain = 'spectrum.library.concordia.ca'

    rp = urllib.robotparser.RobotFileParser() # Set up robots.txt parser
    rp.set_url(urljoin(base_url, '/robots.txt'))
    rp.read()

    while urls_to_visit and len(thesis_links) < max_thesis:
        current_url = urls_to_visit.pop(0)

        # Check if the URL is allowed by robots.txt
        if not rp.can_fetch('*', current_url):
            continue  # Skip disallowed URLs

        if current_url in visited_urls:
            continue
        visited_urls.add(current_url)

        try:
            response = requests.get(current_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            
            for a_tag in soup.find_all('a', href=True):# Find all links on the page
                href = a_tag['href']
                full_link = urljoin(current_url, href)
                parsed_link = urlparse(full_link)

                if domain not in parsed_link.netloc:
                    continue

                full_link = parsed_link.scheme + '://' + parsed_link.netloc + parsed_link.path

                
                if not rp.can_fetch('*', full_link): # Skip if the URL is disallowed by robots.txt
                    continue

                if full_link in visited_urls or full_link in urls_to_visit:
                    continue

                if "/id/eprint/" in full_link:
                    thesis_links.append(full_link)
                    if len(thesis_links) >= max_thesis:
                        return thesis_links
                else:
                    urls_to_visit.append(full_link)

            time.sleep(0.5)

        except Exception as e:
            continue

    return thesis_links



def get_pdf_and_metadata(thesis_url):
    """
        Fetches the PDF link and metadata from a thesis URL.

        Args:
            thesis_url (str): The URL of the thesis.

        Returns:
            tuple: A tuple containing the PDF link, faculty, and department.
        """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}        
        response = requests.get(thesis_url, headers=headers)
        
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        pdf_tag = soup.find('a', class_='ep_document_link', href=True)
        pdf_link = urljoin(thesis_url, pdf_tag['href']) if pdf_tag else None

        # Extract faculty and department information
        faculty = None
        department = None
        divisions_th = soup.find('th', string="Divisions:")
        if divisions_th:
            td_tag = divisions_th.find_next_sibling('td')
            if td_tag:
                division_links = [a.get_text(strip=True) for a in td_tag.find_all('a', href=True)]
                if len(division_links) > 1:
                    faculty = division_links[1]  
                if len(division_links) > 2:
                    department = division_links[2]  

        return pdf_link, faculty, department
    except Exception as e:
        print(f"Error fetching metadata from {thesis_url}: {e}")

    return None, None, None

count = 0


def extract_text_from_pdf(pdf_url):
    """
        Extracts text content from a PDF URL.

        Args:
            pdf_url (str): The URL of the PDF.

        Returns:
            str: The extracted text content.
    """
    try:

        response = requests.get(pdf_url)
        response.raise_for_status()

        # Load the PDF content in memory
        pdf_data = BytesIO(response.content)
        reader = PyPDF2.PdfReader(pdf_data)

        # Extract text from all pages except the first
        text = ""
        for page_num in range(1, len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

        return text
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None



def tokenize_text(text):
    """
    Tokenizes and normalizes text using NLTK.

    Args:
        text (str): The input text.

    Returns:
        list: A list of normalized tokens.
    """
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {"doi", "et", "al", "vol", "author", "abstract", "summary", "introduction", "background", "conclusion", "discussion",
                         "methodology", "methods", "results", "analysis", "findings", "literature", "review", "theory", "experiment", "experiments",
                         "study", "studies", "research", "data", "evidence", "chapter", "section", "figure", "figures", "table", "tables",
                         "appendix", "appendices", "references", "bibliography", "acknowledgments", "preface"}
    
    all_stop_words = stop_words.union(custom_stop_words)

    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(text)

    normalized_tokens = []
    for token in tokens:
        token = token.lower()
        token = token.translate(str.maketrans('', '', string.punctuation))

        if any(char.isdigit() for char in token):
            continue

        if token and token not in all_stop_words and len(token) > 1:
            normalized_token = lemmatizer.lemmatize(token)
            normalized_tokens.append(normalized_token)

    return normalized_tokens

def add_to_inverted_index(inverted_index, tokens, doc_id):
    """
    Add tokens to the inverted index with their positions.
    """
    for position, token in enumerate(tokens):
        if token not in inverted_index:
            inverted_index[token] = defaultdict(list)
        inverted_index[token][doc_id].append(position)


def create_document_collection(inverted_index):
    """
    Converts an inverted index into a document collection.

    Args:
        inverted_index (dict): The inverted index.

    Returns:
        dict: A dictionary where each key is a document ID and its value is a string of tokens.
    """
    doc_collection = defaultdict(list)
    for token, doc_positions in inverted_index.items():
        for doc_id, positions in doc_positions.items():
            if token:
                doc_collection[doc_id].extend([token] * len(positions))  # Add token multiple times for its frequency

    return {doc_id: " ".join(filter(None, tokens)) for doc_id, tokens in doc_collection.items()}

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def perform_clustering(doc_collection, k):
    """
    Perform KMeans clustering on the document collection using TF-IDF.

    Args:
        doc_collection (dict): A dictionary where each key is a document ID and its value is a string of tokens.
        k (int): The number of clusters.

    Returns:
        tuple: (cluster labels, top terms, document-term matrix, document IDs, vectorizer)
    """
    
    vectorizer = TfidfVectorizer()
    doc_term_matrix = vectorizer.fit_transform(doc_collection.values())

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(doc_term_matrix)

    cluster_labels = kmeans.labels_
    feature_names = vectorizer.get_feature_names_out()
    top_terms = {}

    # Extract top 50 terms for each cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    for cluster_id in range(k):
        top_terms[cluster_id] = [feature_names[i] for i in order_centroids[cluster_id, :50]]

    doc_ids = list(doc_collection.keys())

    return cluster_labels, top_terms, doc_term_matrix, doc_ids, vectorizer



def cluster_and_analyze(doc_collection, k, label):
    """
    Perform clustering and output summarized top terms for each cluster.

    Args:
        doc_collection (dict): The document collection.
        k (int): The number of clusters.
        label (str): The label for clustering results.

    Returns:
        dict: A summary of top terms for each cluster.
    """
    cluster_labels, top_terms, doc_term_matrix, doc_ids, vectorizer = perform_clustering(doc_collection, k)

    top_term_summary = {}
    for cluster_id in range(k):
        terms = top_terms[cluster_id][:10]  # Show only the top 10 terms
        top_term_summary[cluster_id] = terms

    print(f"\nSummary of Top Terms for {label} Clustering:")
    for cluster_id, terms in top_term_summary.items():
        print(f"Cluster {cluster_id}: {', '.join(terms)}")

    return top_term_summary 


def visualize_clusters_pca(doc_term_matrix, cluster_labels, title="PCA Cluster Visualization"):
    """
    Visualize clusters using PCA (no saving, just displaying the plot).

    Args:
        doc_term_matrix (scipy.sparse matrix): The document-term matrix.
        cluster_labels (list): Cluster labels for each document.
        title (str): Title for the plot.

    Returns:
        None
    """
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(doc_term_matrix.toarray())

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_labels, cmap="viridis", alpha=0.8, s=50)
    
    legend = plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    plt.gca().add_artist(legend)
    plt.title(title, fontsize=16)
    plt.xlabel("PCA Component 1", fontsize=12)
    plt.ylabel("PCA Component 2", fontsize=12)
    plt.show() 



def fetch_and_process_theses(number_of_thesis):
    """
    Fetches theses, extracts text, tokenizes, and builds an inverted index.

    Args:
        number_of_thesis (int): The number of theses to process.

    Returns:
        tuple: (doc_collection, thesis_metadata, faculties, departments)
    """
    BASE_URL = "https://spectrum.library.concordia.ca/"
    inverted_index = defaultdict(lambda: defaultdict(list))
    thesis_metadata = {} 
    MAX_THESIS = number_of_thesis  

    print("Fetching thesis links...")
    thesis_links = get_thesis_links(BASE_URL, MAX_THESIS)
    print(f"Total number of thesis links collected: {len(thesis_links)}")

    if not thesis_links:
        print("No thesis links found. Exiting.")
        return None, None, None, None

    print(f"Processing {len(thesis_links)} thesis links...")
    processed_count = 0
    doc_id = 1 

    for thesis_url in thesis_links:
        try:
            pdf_link, faculty, department = get_pdf_and_metadata(thesis_url)
            if pdf_link:
                text = extract_text_from_pdf(pdf_link)
                if text:
                    tokens = tokenize_text(text)
                    add_to_inverted_index(inverted_index, tokens, doc_id)
                    thesis_metadata[doc_id] = {
                        "faculty": faculty,
                        "department": department
                    }
                    processed_count += 1
                    doc_id += 1
                else:
                    continue
            else:
                print(f"No PDF link found for {thesis_url}. Skipping.")
        except Exception:
            continue

    if processed_count == 0:
        print("No theses were processed successfully. Exiting.")
        return None, None, None, None

    print("\nBuilding document collection for clustering...")
    doc_collection = create_document_collection(inverted_index)

    faculties = set(md["faculty"] for md in thesis_metadata.values() if md["faculty"])
    departments = set(md["department"] for md in thesis_metadata.values() if md["department"])

    return doc_collection, thesis_metadata, faculties, departments


def perform_clustering_analysis(doc_collection, faculties, departments, cluster_list):
    """
    Perform clustering based on faculties and departments.

    Args:
        doc_collection (dict): The document collection.
        faculties (set): Set of faculties found in metadata.
        departments (set): Set of departments found in metadata.
        cluster_list (list): A list of numbers representing cluster sizes.
    """
    if not doc_collection:
        print("No document collection available for clustering.")
        return

    if departments:
        print("\nClustering based on number of departments...")
        num_departments = len(departments)
        dept_labels, dept_top_terms, dept_doc_term_matrix, _ = perform_clustering(doc_collection, num_departments)
        print(f"Departments extracted: {', '.join(departments)}")

        sample_silhouette_values = silhouette_samples(dept_doc_term_matrix, dept_labels)
        unique_clusters = np.unique(dept_labels)
        cluster_silhouette_scores = {cluster: sample_silhouette_values[np.where(dept_labels == cluster)].mean()
                                     for cluster in unique_clusters}
        sorted_clusters = sorted(cluster_silhouette_scores.items(), key=lambda x: x[1], reverse=True)

        print("\nTop department clusters:")
        for cluster_id, _ in sorted_clusters:
            print(f"Cluster {cluster_id} - Top 50 terms: {', '.join(dept_top_terms[cluster_id])}")

    if faculties:
        print("\nClustering based on number of faculties...")
        num_faculties = len(faculties)
        fac_labels, fac_top_terms, fac_doc_term_matrix, _ = perform_clustering(doc_collection, num_faculties)
        print(f"Faculties extracted: {', '.join(faculties)}")

        sample_silhouette_values = silhouette_samples(fac_doc_term_matrix, fac_labels)
        unique_clusters = np.unique(fac_labels)
        cluster_silhouette_scores = {cluster: sample_silhouette_values[np.where(fac_labels == cluster)].mean()
                                     for cluster in unique_clusters}
        sorted_clusters = sorted(cluster_silhouette_scores.items(), key=lambda x: x[1], reverse=True)

        print("\nTop faculty clusters:")
        for cluster_id, _ in sorted_clusters:
            print(f"Cluster {cluster_id} - Top 50 terms: {', '.join(fac_top_terms[cluster_id])}")
            

def perform_nlp_analysis(doc_collection, n_topics=5, n_top_words=10, max_docs=5, ratio=0.2):
    """
    Perform NLP tasks: Topic Modeling (LDA/NMF), Named Entity Recognition (NER), and Text Summarization.

    Args:
        doc_collection (dict): The document collection.
        n_topics (int): Number of topics for LDA/NMF.
        n_top_words (int): Number of words per topic.
        max_docs (int): Number of documents for NER & summarization.
        ratio (float): Percentage of sentences to keep in summaries.

    Returns:
        dict: Combined results from all NLP tasks.
    """
    if not doc_collection:
        print("No document collection available for NLP analysis.")
        return

    _, _, doc_term_matrix, _, vectorizer = perform_clustering(doc_collection, k=n_topics)

    lda_topics = perform_lda_topic_modeling(doc_term_matrix, vectorizer, n_topics=n_topics, n_top_words=n_top_words)
    nmf_topics = perform_nmf_topic_modeling(doc_term_matrix, vectorizer, n_topics=n_topics, n_top_words=n_top_words)
    named_entities = extract_named_entities_nltk(doc_collection, max_docs=max_docs)
    summaries = summarize_documents(doc_collection, doc_term_matrix, vectorizer, max_docs=max_docs, ratio=ratio)

    return {
        "lda_topics": lda_topics,
        "nmf_topics": nmf_topics,
        "named_entities": named_entities,
        "summaries": summaries,
    }

if __name__ == "__main__":
    number_of_thesis = 50
    
    doc_collection, thesis_metadata, faculties, departments = fetch_and_process_theses(number_of_thesis)

    if doc_collection:
        #clustering_results = perform_clustering_analysis(doc_collection, faculties, departments, cluster_list)
        nlp_results = perform_nlp_analysis(doc_collection)

        print("\n===== Summary of Results =====")
        print("\nKey Topics (LDA & NMF):", list(nlp_results["lda_topics"].keys()))
        print("\nNamed Entities Distribution:", nlp_results["named_entities"])    