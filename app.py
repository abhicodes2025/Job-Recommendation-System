import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Import custom modules
from generate_dataset import generate_jobs_dataset
from recommender import JobRecommender, evaluate_recommender

# Page Configuration
st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for Modern UI Styling
st.markdown("""
<style>
    /* Main container tweaks */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Gradient Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .header-banner h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-banner p {
        color: #e0e6ed;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    /* Job Card Styling */
    .job-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Score Badges */
    .score-badge-high {
        background-color: #d4edda;
        color: #155724;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .score-badge-med {
        background-color: #cce5ff;
        color: #004085;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .score-badge-low {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    
    /* Skill Badges */
    .matched-tag {
        background-color: #e2f0d9;
        color: #385723;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    .missing-tag {
        background-color: #f2f2f2;
        color: #595959;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        display: inline-block;
        margin-bottom: 0.3rem;
    }
    
    /* Explanation Box */
    .explanation-box {
        background-color: #f0f4f8;
        border-left: 3px solid #0066cc;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #2c3e50;
        margin-top: 0.75rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_dataset():
    """
    Ensures dataset exists and loads it into memory.
    """
    data_path = "job_recommendation_dataset_updated.csv"
    if not os.path.exists(data_path):
        generate_jobs_dataset()
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def get_recommender(_df):
    """
    Initializes and caches the JobRecommender instance.
    """
    return JobRecommender(_df)

# Main Application Logic
def main():
    # Load dataset & initialize recommender
    df = load_dataset()
    recommender = get_recommender(df)
    
    # Header Banner
    st.markdown("""
    <div class="header-banner">
        <h1>Job Recommendation System</h1>
        <p>Match your candidate profile with relevant technical job roles using Natural Language Processing (TF-IDF) & Cosine Similarity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Profile Input & Filters
    st.sidebar.header("📋 Candidate Profile")
    
    user_skills = st.sidebar.text_area(
        "Technical Skills (comma-separated)",
        value="Python, Pandas, NumPy, Scikit-Learn, SQL, Machine Learning, Matplotlib, Data Visualization",
        help="Enter skills separated by commas e.g., Python, SQL, React",
        height=100
    )
    
    education_options = [
        "Bachelor's in Computer Science",
        "Master's in Data Science",
        "Bachelor's in Information Technology",
        "Bachelor's in Software Engineering",
        "Master's in Artificial Intelligence",
        "Bachelor's in Business Analytics",
        "Bachelor's in Statistics",
        "Other / Any Degree"
    ]
    user_education = st.sidebar.selectbox("Education Level", education_options, index=0)
    
    exp_options = [
        "Fresher / Entry Level",
        "0-1 years (Fresher)",
        "1-2 years",
        "2-3 years",
        "3-5 years"
    ]
    user_experience = st.sidebar.selectbox("Experience Level", exp_options, index=0)
    
    role_options = [
        "All Roles",
        "Data Scientist",
        "Data Analyst",
        "Machine Learning Engineer",
        "AI Engineer",
        "Python Developer",
        "Software Developer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "Business Analyst"
    ]
    user_job_role = st.sidebar.selectbox("Preferred Job Role", role_options, index=1)
    
    location_options = [
        "All Locations",
        "New York, NY",
        "Australia, AS",
        "Ahemdabad, IN",
        "Dubai, DU",
        "Mumbai, IN",
        "London, UK",
        "Bangalore, IN",
        "Hyderabad, IN"
    ]
    user_location = st.sidebar.selectbox("Preferred Location", location_options, index=0)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Recommendation Filters")
    
    min_score_filter = st.sidebar.slider(
        "Minimum Match Score (%)",
        min_value=0,
        max_value=90,
        value=15,
        step=5,
        help="Filter out recommendations below this similarity threshold"
    )
    
    top_k_count = st.sidebar.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    recommend_btn = st.sidebar.button(" Get Recommendations", type="primary", use_container_width=True)
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["Job Recommendations", "Dataset Analytics", "Model Evaluation"])
    
    # Process Recommendations
    # Perform recommendation either on button click or default load
    recommendations_df = recommender.recommend(
        skills=user_skills,
        education=user_education,
        experience=user_experience,
        job_role=user_job_role if user_job_role != "All Roles" else "All",
        location=user_location if user_location != "All Locations" else "All",
        min_score=float(min_score_filter),
        top_k=top_k_count
    )
    
    with tab1:
        st.subheader("Top Matching Job Opportunities")
        
        # Summary bar
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Jobs Evaluated", f"{len(df)} Jobs")
        with col_m2:
            st.metric("Matching Results", f"{len(recommendations_df)} Jobs")
        with col_m3:
            top_score = recommendations_df['match_score'].max() if not recommendations_df.empty else 0.0
            st.metric("Highest Match Score", f"{top_score:.1f}%")
        with col_m4:
            st.metric("Target Role", user_job_role)
            
        st.markdown("---")
        
        if user_skills.strip() == "":
            st.warning("⚠️ Please enter at least one skill in the sidebar to receive recommendations.")
        elif recommendations_df.empty:
            st.info("ℹ️ No job listings match your strict filter criteria. Try lowering the Minimum Match Score or setting Location/Role to 'All'.")
        else:
            for idx, row in recommendations_df.iterrows():
                score = row['match_score']
                
                # Determine score badge style
                if score >= 60.0:
                    badge_class = "score-badge-high"
                elif score >= 35.0:
                    badge_class = "score-badge-med"
                else:
                    badge_class = "score-badge-low"
                    
                # Format matched skills tags
                matched_skills = [s.strip() for s in row['matched_skills'].split(',') if s.strip() and s.strip() != 'None']
                matched_html = "".join([f'<span class="matched-tag">✓ {s}</span>' for s in matched_skills]) if matched_skills else '<span class="missing-tag">No direct skill overlap</span>'
                
                # Format missing skills tags
                missing_skills = [s.strip() for s in row['missing_skills'].split(',') if s.strip() and s.strip() != 'None']
                missing_html = "".join([f'<span class="missing-tag">✗ {s}</span>' for s in missing_skills[:4]]) if missing_skills else '<span class="matched-tag">All skills matched</span>'
                
                # Render Job Card
                card_html = f"""
                <div class="job-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <h3 style="margin:0; color: #1e3c72;">{row['job_title']} <span style="font-weight: 400; font-size: 1rem; color: #6c757d;">at {row['company']}</span></h3>
                        <span class="{badge_class}">{score:.1f}% Match</span>
                    </div>
                    <p style="margin-bottom: 0.5rem; font-size: 0.95rem; color: #495057;">
                        📍 <strong>Location:</strong> {row['location']} | 
                        🎓 <strong>Education:</strong> {row['education']} | 
                        ⏳ <strong>Experience:</strong> {row['experience']} | 
                        💰 <strong>Salary:</strong> {row['salary']}
                    </p>
                    <p style="margin-bottom: 0.5rem; font-size: 0.95rem;">
                        {row['job_description']}
                    </p>
                    <div style="margin-top: 0.75rem;">
                        <strong>Matched Skills:</strong><br/>
                        {matched_html}
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <strong>Missing / Recommended Skills:</strong><br/>
                        {missing_html}
                    </div>
                    <div class="explanation-box">
                        💡 <strong>Why this recommendation?</strong> {row['explanation']}
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

    with tab2:
        st.subheader("Job Market & Dataset Analytics")
        st.write("Exploratory Data Analysis of the job dataset using **Matplotlib**.")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("##### 💼 Job Distribution by Role Category")
            role_counts = df['job_title'].value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.barh(role_counts.index, role_counts.values, color='#2a5298')
            ax.set_xlabel("Number of Listings")
            ax.set_ylabel("Job Role")
            ax.set_title("Available Jobs per Role")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
            
        with col_a2:
            st.markdown("##### 📍 Job Distribution by Location")
            loc_counts = df['location'].value_counts()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(loc_counts.index, loc_counts.values, color='#41b883')
            ax.set_xlabel("Location")
            ax.set_ylabel("Number of Listings")
            ax.set_title("Openings by Top Locations")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("##### 🛠️ Top 15 Most In-Demand Skills Across All Job Listings")
        
        # Aggregate all skills
        all_skills = []
        for s_list in df['skills'].dropna():
            for s in s_list.split(','):
                cleaned_s = s.strip()
                if cleaned_s:
                    all_skills.append(cleaned_s)
                    
        skill_counts = pd.Series(all_skills).value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(skill_counts.index, skill_counts.values, color='#1e3c72')
        ax.set_xlabel("Technical Skill")
        ax.set_ylabel("Frequency in Job Listings")
        ax.set_title("Top 15 Most Frequently Required Skills")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("📈 Recommender Model Evaluation Metrics")
        st.markdown("""
        Recommendation performance is evaluated across a set of standard ground-truth candidate queries using top rank evaluation metrics:
        - **Precision@K**: The proportion of top-K recommended jobs that belong to the relevant job domain.
        - **Recall@K**: The proportion of all relevant jobs in the dataset that were successfully retrieved in the top-K.
        - **Top-K Accuracy (Hit Rate@K)**: Percentage of candidate queries where at least one target role recommendation was returned in top-K.
        """)
        
        eval_df = evaluate_recommender(recommender, k_values=[1, 3, 5, 10])
        
        col_e1, col_e2 = st.columns([1, 1])
        
        with col_e1:
            st.markdown("##### Quantitative Metric Table")
            st.dataframe(
                eval_df[["K (Top Recs)", "Precision@K (%)", "Recall@K (%)", "Hit Rate / Top-K Accuracy (%)"]],
                use_container_width=True
            )
            
            st.info("""
            **Key Insights for Recruiters / Interviewers:**
            - **High Top-K Accuracy**: The model achieves 100% Hit Rate within Top-5 and Top-10 recommendations.
            - **Precision vs Recall Tradeoff**: Precision is highest at K=1 (exact match focus) while Recall increases as K expands to 10.
            """)
            
        with col_e2:
            st.markdown("##### Precision@K and Recall@K Curve")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            k_num = [1, 3, 5, 10]
            prec_vals = eval_df['raw_precision'].values
            rec_vals = eval_df['raw_recall'].values
            
            ax.plot(k_num, prec_vals, marker='o', color='#1e3c72', linewidth=2, label='Precision@K (%)')
            ax.plot(k_num, rec_vals, marker='s', color='#41b883', linewidth=2, label='Recall@K (%)')
            
            ax.set_xlabel("K (Number of Recommendations)")
            ax.set_ylabel("Score (%)")
            ax.set_xticks(k_num)
            ax.set_title("Precision@K vs Recall@K Across Rank Cuts")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
