import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Cleans raw input text for NLP processing.
    
    Steps:
    1. Handle non-string / null values gracefully.
    2. Convert text to lowercase.
    3. Remove unnecessary special characters (preserving spaces, numbers, and key symbols like +, #).
    4. Normalize whitespace (stripping extra spaces).
    
    Parameters:
        text (str): Raw input text.
        
    Returns:
        str: Preprocessed clean text.
    """
    if not isinstance(text, str) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove unnecessary special characters but keep letters, numbers, spaces, and tech symbols like +, #
    text = re.sub(r'[^a-z0-9\s+#]', ' ', text)
    
    # Normalize multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the job dataset DataFrame:
    1. Fills missing values with empty strings.
    2. Removes duplicate records based on job_id and core attributes.
    3. Combines relevant text fields (job_title, skills, education, experience, job_description).
    4. Applies clean_text to the combined text feature column.
    
    Parameters:
        df (pd.DataFrame): Raw jobs DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with an added 'cleaned_combined_text' column.
    """
    # Make a copy to avoid mutating original data inplace unintentionally
    df = df.copy()
    
    # Step 1: Handle missing values safely
    text_columns = ['job_title', 'company', 'location', 'skills', 'experience', 'education', 'job_description', 'salary']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
            
    # Step 2: Remove duplicate records
    if 'job_id' in df.columns:
        df = df.drop_duplicates(subset=['job_id'])
    else:
        df = df.drop_duplicates(subset=['job_title', 'company', 'location'])
        
    # Step 3: Combine relevant text fields into a single text representation
    # Combining title, skills, education, experience, and description allows TF-IDF to capture full semantic context
    df['combined_text'] = (
        df['job_title'] + " " +
        df['skills'] + " " +
        df['education'] + " " +
        df['experience'] + " " +
        df['job_description']
    )
    
    # Step 4: Clean the combined text
    df['cleaned_combined_text'] = df['combined_text'].apply(clean_text)
    
    return df

def preprocess_user_input(skills: str, education: str, experience: str, job_role: str, location: str, job_prefs: str = "") -> str:
    """
    Combines and cleans user profile inputs into a single query string for recommendation matching.
    
    Parameters:
        skills (str): User skills (e.g. "Python, Pandas, SQL, ML")
        education (str): User education background (e.g. "Bachelor's in Computer Science")
        experience (str): User experience (e.g. "1 year")
        job_role (str): Preferred job role (e.g. "Data Scientist")
        location (str): Preferred location (e.g. "Remote")
        job_prefs (str): Additional job preferences or notes
        
    Returns:
        str: Cleaned unified user profile text.
    """
    raw_user_profile = f"{job_role} {skills} {education} {experience} {location} {job_prefs}"
    return clean_text(raw_user_profile)

if __name__ == "__main__":
    # Simple self-test
    sample_text = "Python, Pandas & SQL!  Experience: 2+ Years #MachineLearning"
    cleaned = clean_text(sample_text)
    print("Sample Raw Text:", sample_text)
    print("Sample Cleaned:", cleaned)
