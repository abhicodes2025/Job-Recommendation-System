import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import clean_text, preprocess_user_input, preprocess_dataframe

class JobRecommender:
    """
    Job Recommendation Engine using TF-IDF Vectorization and Cosine Similarity.
    """
    def __init__(self, jobs_df: pd.DataFrame):
        """
        Initializes the recommender with the jobs DataFrame and trains the TF-IDF Vectorizer.
        """
        self.raw_df = jobs_df
        self.df = preprocess_dataframe(jobs_df)
        
        # Initialize TF-IDF Vectorizer with unigrams & bigrams and English stop words
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        
        # Compute TF-IDF matrix for all job listings
        self.job_tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_combined_text'])

    def recommend(
        self,
        skills: str,
        education: str,
        experience: str,
        job_role: str,
        location: str = "All",
        min_score: float = 0.0,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Recommends the top K jobs based on similarity between user profile and job listings.
        
        Parameters:
            skills (str): User skills (comma-separated or string)
            education (str): User education level
            experience (str): User experience level
            job_role (str): Preferred job role
            location (str): Desired location filter ("All" or specific location)
            min_score (float): Minimum match score percentage (0 - 100)
            top_k (int): Number of top recommendations to return (default 10)
            
        Returns:
            pd.DataFrame: Top K recommended jobs with match scores, skill matching, and explanations.
        """
        # Preprocess user input
        cleaned_user_query = preprocess_user_input(
            skills=skills,
            education=education,
            experience=experience,
            job_role=job_role,
            location=location if location != "All" else ""
        )
        
        if not cleaned_user_query.strip():
            # If query is completely empty, return empty dataframe with required schema
            result_df = self.df.head(0).copy()
            result_df['match_score'] = []
            result_df['matched_skills'] = []
            result_df['missing_skills'] = []
            result_df['explanation'] = []
            return result_df

        # Vectorize user profile query
        user_vector = self.vectorizer.transform([cleaned_user_query])
        
        # Compute cosine similarity between user query vector and all job vectors
        similarity_scores = cosine_similarity(user_vector, self.job_tfidf_matrix).flatten()
        
        # Copy original dataframe to add scores and analysis
        results = self.df.copy()
        
        # Convert score to percentage (0.0 to 100.0)
        results['match_score'] = np.round(similarity_scores * 100, 1)
        
        # Parse user skills list for skill gap matching
        user_skill_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
        
        matched_skills_list = []
        missing_skills_list = []
        explanations = []
        
        for idx, row in results.iterrows():
            job_skills = [s.strip() for s in row['skills'].split(',') if s.strip()]
            job_skills_lower = [s.lower() for s in job_skills]
            
            # Find matched skills and missing skills
            matched = []
            missing = []
            
            for original_skill, lower_skill in zip(job_skills, job_skills_lower):
                # Check if any user skill matches job skill (exact or substring match)
                if any(u_skill in lower_skill or lower_skill in u_skill for u_skill in user_skill_list if len(u_skill) > 1):
                    matched.append(original_skill)
                else:
                    missing.append(original_skill)
                    
            matched_str = ", ".join(matched) if matched else "None"
            missing_str = ", ".join(missing[:4]) if missing else "None"  # limit display to top 4 missing
            
            matched_skills_list.append(matched_str)
            missing_skills_list.append(missing_str)
            
            # Generate explanation
            explanation = self._generate_explanation(
                job_title=row['job_title'],
                matched_skills=matched,
                match_score=row['match_score']
            )
            explanations.append(explanation)
            
        results['matched_skills'] = matched_skills_list
        results['missing_skills'] = missing_skills_list
        results['explanation'] = explanations
        
        # Filtering logic
        if location and location != "All":
            results = results[results['location'].str.lower() == location.lower()]
            
        if job_role and job_role != "All":
            # Flexible filtering by role title
            results = results[results['job_title'].str.lower().str.contains(job_role.lower(), regex=False)]
            
        if min_score > 0:
            results = results[results['match_score'] >= min_score]
            
        # Sort by match score descending
        results = results.sort_values(by='match_score', ascending=False)
        
        return results.head(top_k)

    def _generate_explanation(self, job_title: str, matched_skills: list, match_score: float) -> str:
        """
        Generates a concise, natural language explanation for why a job was recommended.
        """
        if matched_skills:
            skills_phrase = ", ".join(matched_skills[:3])
            if len(matched_skills) > 3:
                skills_phrase += f" and {len(matched_skills) - 3} other skills"
            return f"Recommended because your {skills_phrase} skill(s) closely match the requirements for this {job_title} role."
        elif match_score > 40.0:
            return f"Recommended due to strong overall alignment in your profile, education, and preferred job role."
        else:
            return f"Moderate match based on general keywords in your profile and experience."

def evaluate_recommender(recommender: JobRecommender, k_values: list = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Evaluates the recommendation system across a set of ground-truth test queries.
    
    Metrics evaluated:
    - Precision@K: Fraction of top-K recommendations that belong to the query's target domain.
    - Recall@K: Fraction of relevant jobs in the dataset retrieved within top-K.
    - Hit Rate@K (Top-K Accuracy): Proportion of queries where at least 1 top-K job is relevant.
    """
    test_cases = [
        {
            "query_role": "Data Scientist",
            "skills": "Python, Pandas, NumPy, Scikit-Learn, Machine Learning, SQL, Statistics",
            "education": "Master's in Data Science",
            "experience": "2 years",
            "target_role": "Data Scientist"
        },
        {
            "query_role": "Data Analyst",
            "skills": "SQL, Excel, Python, Pandas, Power BI, Tableau, Reporting",
            "education": "Bachelor's in Business Analytics",
            "experience": "1 year",
            "target_role": "Data Analyst"
        },
        {
            "query_role": "Machine Learning Engineer",
            "skills": "Python, TensorFlow, PyTorch, Scikit-Learn, Deep Learning, Docker, SQL",
            "education": "Master's in Machine Learning",
            "experience": "3 years",
            "target_role": "Machine Learning Engineer"
        },
        {
            "query_role": "AI Engineer",
            "skills": "Python, PyTorch, NLP, Generative AI, LangChain, Transformers",
            "education": "Master's in AI",
            "experience": "2 years",
            "target_role": "AI Engineer"
        },
        {
            "query_role": "Python Developer",
            "skills": "Python, Django, Flask, FastAPI, SQL, PostgreSQL, REST APIs, Git",
            "education": "Bachelor's in Computer Science",
            "experience": "2 years",
            "target_role": "Python Developer"
        },
        {
            "query_role": "Frontend Developer",
            "skills": "JavaScript, TypeScript, React, HTML5, CSS3, Tailwind CSS, Redux",
            "education": "Bachelor's in Web Development",
            "experience": "2 years",
            "target_role": "Frontend Developer"
        },
        {
            "query_role": "Backend Developer",
            "skills": "Node.js, Express, PostgreSQL, MongoDB, REST APIs, Microservices, Python",
            "education": "Bachelor's in Computer Science",
            "experience": "3 years",
            "target_role": "Backend Developer"
        },
        {
            "query_role": "Full Stack Developer",
            "skills": "Python, JavaScript, React, Node.js, SQL, MongoDB, HTML5, CSS3",
            "education": "Bachelor's in Computer Science",
            "experience": "2 years",
            "target_role": "Full Stack Developer"
        },
        {
            "query_role": "Business Analyst",
            "skills": "Business Analysis, SQL, Excel, Data Modeling, Requirements Gathering, Jira",
            "education": "Bachelor's in Business Administration",
            "experience": "2 years",
            "target_role": "Business Analyst"
        },
        {
            "query_role": "Software Developer",
            "skills": "Python, Java, C++, Data Structures, Algorithms, Git, Object Oriented Programming",
            "education": "Bachelor's in Computer Engineering",
            "experience": "1 year",
            "target_role": "Software Developer"
        }
    ]
    
    evaluation_results = []
    
    for k in k_values:
        precisions = []
        recalls = []
        hits = []
        
        for case in test_cases:
            target_role = case['target_role']
            total_relevant_in_db = len(recommender.df[recommender.df['job_title'] == target_role])
            if total_relevant_in_db == 0:
                continue
                
            recs = recommender.recommend(
                skills=case['skills'],
                education=case['education'],
                experience=case['experience'],
                job_role=case['query_role'],
                location="All",
                min_score=0.0,
                top_k=k
            )
            
            # Count relevant recommendations in top K
            relevant_retrieved = sum(1 for title in recs['job_title'] if target_role.lower() in title.lower())
            
            precision_k = relevant_retrieved / k
            recall_k = relevant_retrieved / total_relevant_in_db
            hit_k = 1.0 if relevant_retrieved > 0 else 0.0
            
            precisions.append(precision_k)
            recalls.append(recall_k)
            hits.append(hit_k)
            
        avg_precision = np.mean(precisions) * 100 if precisions else 0
        avg_recall = np.mean(recalls) * 100 if recalls else 0
        hit_rate = np.mean(hits) * 100 if hits else 0
        
        evaluation_results.append({
            "K (Top Recs)": f"Top-{k}",
            "Precision@K (%)": f"{avg_precision:.1f}%",
            "Recall@K (%)": f"{avg_recall:.1f}%",
            "Hit Rate / Top-K Accuracy (%)": f"{hit_rate:.1f}%",
            "raw_precision": avg_precision,
            "raw_recall": avg_recall,
            "raw_hit": hit_rate
        })
        
    return pd.DataFrame(evaluation_results)

if __name__ == "__main__":
    # Quick functional test
    from generate_dataset import generate_jobs_dataset
    import os
    
    if not os.path.exists("jobs.csv"):
        generate_jobs_dataset()
        
    df = pd.read_csv("jobs.csv")
    recommender = JobRecommender(df)
    
    sample_recs = recommender.recommend(
        skills="Python, Pandas, SQL, Scikit-Learn, Machine Learning",
        education="Bachelor's in Computer Science",
        experience="1 year",
        job_role="Data Scientist",
        top_k=5
    )
    
    print("\n--- Top 5 Sample Recommendations ---")
    for idx, row in sample_recs.iterrows():
        print(f"[{row['job_id']}] {row['job_title']} at {row['company']} | Score: {row['match_score']}%")
        print(f"   Matched Skills: {row['matched_skills']}")
        print(f"   Explanation: {row['explanation']}\n")
        
    print("\n--- Evaluation Metrics ---")
    eval_df = evaluate_recommender(recommender)
    print(eval_df[["K (Top Recs)", "Precision@K (%)", "Recall@K (%)", "Hit Rate / Top-K Accuracy (%)"]])
