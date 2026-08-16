# Job Recommendation System

A complete, portfolio-ready **Job Recommendation System** built using **Python**, **Pandas**, **Natural Language Processing (NLP)**, **TF-IDF Vectorization**, **Cosine Similarity**, **Streamlit**, and **Matplotlib**. 

This application analyzes candidate profiles (skills, education, experience, preferred role, location) and matches them against job descriptions using vector space modeling to deliver top-ranked relevant job recommendations with match percentages, skill gap analysis, and dynamic explanations.

---

## Project Overview & Problem Statement

### **Problem Statement**
Job seekers often spend hours scrolling through hundreds of generic job listings that do not match their exact skill sets or qualifications. Traditional keyword-search systems often miss semantic matches or fail to rank postings by similarity.

### **Solution**
This project builds a content-based recommendation engine that treats candidate profiles and job listings as numerical term-vectors in a high-dimensional TF-IDF feature space. By computing **Cosine Similarity**, the system ranks all available job listings by relevance, highlighting matched skills, missing skills, and dynamic rationale for every match.

---

## Features

- 👤 **Interactive Candidate Profile Builder**: Inputs for technical skills, education level, experience level, preferred role, and preferred location.
-  **Content-Based NLP Recommendation Engine**: Utilizes TF-IDF vectorization with unigrams & bigrams and Cosine Similarity.
-  **Match Percentage Scoring**: Displays recommendation scores as clean percentages (e.g. `91.4% match`).
-  **Skill Matching & Gap Analysis**: Highlights **Matched Skills** in green and **Missing Skills** in grey for candidate upskilling.
-  **Dynamic Natural Language Explanations**: Generates clear rationale explaining *why* a job was recommended.
-  **Interactive Filtering**: Search and filter by preferred location, target job role, and minimum match score threshold.
-  **Market Analytics Dashboard**: Visualizes job role distribution, location breakdown, and top in-demand skills using **Matplotlib**.
-  **Recommendation Metrics Evaluation**: Measures system quality using `Precision@K`, `Recall@K`, and `Top-K Accuracy`.

---

## Technology Stack

| Technology | Category | Usage |
| :--- | :--- | :--- |
| **Python 3.9+** | Core Language | Application logic & machine learning pipeline |
| **Pandas** | Data Processing | Data cleaning, feature aggregation, filtering |
| **NumPy** | Numerical Computing | Vector math and score formatting |
| **Scikit-Learn** | Machine Learning | `TfidfVectorizer` and `cosine_similarity` |
| **Streamlit** | Web Framework | Interactive web application interface |
| **Matplotlib** | Data Visualization | Market analytics & evaluation charts (No Seaborn) |

---

## How the Recommendation System Works

```
                     ┌────────────────────────┐
                     │ Raw Job Dataset        │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ NLP Preprocessing      │
                     │ (Lowercase, Clean,     │
                     │ Combine Features)      │
                     └───────────┬────────────┘
                                 ▼
                     ┌────────────────────────┐
                     │ TF-IDF Vectorization   │
                     │ (Job Vectors Matrix)   │
                     └───────────┬────────────┘
                                 │
  ┌──────────────────────┐       │
  │ User Profile Input   │       │
  └──────────┬───────────┘       │
             ▼                   │
  ┌──────────────────────┐       │
  │ NLP Preprocessing    │       │
  └──────────┬───────────┘       │
             ▼                   │
  ┌──────────────────────┐       │
  │ TF-IDF Transform     │       │
  │ (User Vector)        │       │
  └──────────┬───────────┘       │
             │                   │
             └─────────┬─────────┘
                       ▼
            ┌─────────────────────┐
            │ Cosine Similarity   │
            │ Score Calculation   │
            └──────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │ Ranking & Filtering │
            │ (Top-K Recs & Tags) │
            └─────────────────────┘
```

### 1. **NLP Text Preprocessing** (`preprocessing.py`)
- Missing value handling (`df.fillna('')`) and duplicate removal (`drop_duplicates`).
- Convert text to lowercase.
- Remove non-alphanumeric special characters while retaining key tech symbols (`+`, `#`).
- Normalize extra whitespace.
- Combine text columns (`job_title` + `skills` + `education` + `experience` + `job_description`) into a single text representation `cleaned_combined_text`.
- Clean user profile inputs using the exact same pipeline for feature symmetry.

### 2. **TF-IDF Vectorization**
Term Frequency-Inverse Document Frequency (TF-IDF) converts text into numerical vectors where word weights reflect importance:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

- **Term Frequency ($\text{TF}$)**: Measures how frequently term $t$ appears in job description $d$.
- **Inverse Document Frequency ($\text{IDF}$)**: Penalizes common filler words across all listings $D$.
- Uses unigrams and bigrams (`ngram_range=(1,2)`) to capture multi-word skills like `"machine learning"`, `"data science"`, and `"full stack"`.

### 3. **Cosine Similarity**
Measures the angle between the user profile vector $\vec{U}$ and each job vector $\vec{J}_i$:

$$\text{Similarity}(\vec{U}, \vec{J}_i) = \frac{\vec{U} \cdot \vec{J}_i}{\|\vec{U}\| \|\vec{J}_i\|}$$

The resulting cosine score ranges from $0$ to $1$, which is multiplied by $100$ to produce intuitive percentage match scores (e.g. `88.5% match`).

---

## Evaluation Metrics Explained

Unlike regression models that use $R^2$ or MSE, recommendation systems are evaluated using ranking quality metrics:

- **Precision@K**: The percentage of top-K recommended jobs that belong to the query's target domain.
  $$\text{Precision@K} = \frac{\text{Relevant Jobs in Top } K}{K}$$
- **Recall@K**: The fraction of all relevant jobs in the dataset that were successfully retrieved in the top-K recommendations.
  $$\text{Recall@K} = \frac{\text{Relevant Jobs in Top } K}{\text{Total Relevant Jobs in Dataset}}$$
- **Top-K Accuracy (Hit Rate@K)**: The proportion of test queries for which at least one relevant job was successfully returned within the top-K recommendations.

---

## Installation & How to Run

### **Prerequisites**
- Python 3.9 or higher
- Git & VS Code

### **Step 1: Clone or Navigate to the Workspace**
```bash
cd "c:\Users\anshp\OneDrive\Desktop\Job Recommendation System"
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Run the Streamlit Application**
```bash
streamlit run app.py
```

The application will launch automatically in your web browser at `http://localhost:8501`.

---

## Example Usage

1. Open the Streamlit web app in your browser.
2. In the sidebar under **Candidate Profile**:
   - **Skills**: `Python, SQL, Pandas, NumPy, Scikit-Learn, Machine Learning, Matplotlib`
   - **Education**: `Bachelor's in Computer Science`
   - **Experience**: `Fresher / Entry Level`
   - **Preferred Role**: `Data Scientist`
   - **Location**: `All Locations`
3. Set **Minimum Match Score** to `15%`.
4. View the top 10 recommended jobs with match percentages, matched skills, missing skills, and rationale explanations!

---

## Future Improvements

- **Word Embeddings / Transformers**: Incorporate SBERT (Sentence-BERT) or OpenAI embeddings for deep semantic similarity.
- **Resume Parser**: Allow users to upload PDF/DOCX resumes for automatic skill extraction using PyPDF2 / pdfplumber.
- **Database Integration**: Connect to SQLite or PostgreSQL for real-time job posting updates.
- **Job Alerts**: Send automated email notifications when a job matching $>80\%$ score is posted.

---

## Skills Demonstrated (Resume Highlights)

- **Machine Learning & NLP**: TF-IDF, Vector Space Modeling, Cosine Similarity, Text Mining, N-gram feature extraction.
- **Python Engineering**: Modular OOP design, clean software architecture, caching decorators (`st.cache_data`, `st.cache_resource`).
- **Data Engineering & Analysis**: Pandas DataFrame manipulations, regular expressions text processing, synthetic data generation.
- **Web Development**: Streamlit UI design, interactive dashboards, custom CSS styling.
- **Model Evaluation**: Ranking evaluation metrics (`Precision@K`, `Recall@K`, `Hit Rate`).
