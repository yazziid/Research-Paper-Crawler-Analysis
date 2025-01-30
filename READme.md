# Thesis Clustering and Analysis

This was originally part of my final project for information retrieval. It has been updated to include tools I learned either from the class or myself.  

This project crawls the Concordia Spectrum website to fetch thesis documents, processes their text data, and clusters them based on TF-IDF features. The clusters are visualized using PCA.

---

## Features

- **Web Crawling:** Recursively fetches thesis links and associated metadata.
- **PDF Processing:** Extracts text content from thesis PDFs.
- **Text Normalization:** Tokenizes, lemmatizes, and removes stopwords.
- **Clustering:** Groups theses into clusters using KMeans and visualizes them using PCA.
- **Topic Modeling:** Uses **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)** to extract major research topics.
- **Named Entity Recognition (NER):**  Identifies **organizations, people, and locations** mentioned in the text.
- **Text Summarization:** Uses `sumy`’s **LSA Summarizer** to generate concise summaries of each thesis.

---

## Prerequisites

- Python 3.8+
- A virtual environment (recommended)

---

## Installation

### Set up a Virtual Environment

Create and activate a virtual environment:

#### On Linux/MacOS:

Run:

```bash
python3 -m venv venv
```

Then run:

```bash
source venv/bin/activate
```

#### On Windows:

Run:

```bash
python -m venv venv
```

Then run:

```bash
venv\Scripts\activate
```

### Install Dependencies

Use the provided `requirements.txt` file to install all necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The script is configured to process 50 theses and cluster them into 3 and 6 clusters.

Run the script:

```bash
python RankingCrawler.py
```

---

## Output



---

## Configurations

You can modify the following parameters in the `main` function:

- **`numberthesis`:** Number of theses to process (current: 50).
- **`cluster_list`:** Number of clusters to generate and analyze (default: `[3, 6]`).

---

## Dependencies

The script uses the following Python libraries:


`requests` == 2.32.3
`beautifulsoup4` == 4.12.3
`nltk` == 3.9.1
`scikit-learn` == 1.5.2
`matplotlib` == 3.9.3
`numpy` == 2.1.3
`PyPDF2` == 3.0.1
`sumy` == 0.10.0
`scipy` == 1.11.3
`urllib3` == 2.0.7


Ensure all dependencies are installed using `requirements.txt`.

---

## File Structure

- `RankingCrawler.py`: Main script for crawling, processing, and clustering.
- `someNLP.py`: Contains Topic Modeling, Named Entity Recognition, and Summarization (needs more work)
- `requirements.txt`: Dependencies required to run the script.