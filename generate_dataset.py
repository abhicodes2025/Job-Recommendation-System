import os
import csv
import random
from turtle import st

def generate_jobs_dataset():
    # Target directory structure
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    file_paths = [
        os.path.join(root_dir, "job_recommendation_dataset_updated.csv"),
        os.path.join(data_dir, "job_recommendation_dataset_updated.csv")
    ]
    
    companies_by_location = {

    "New York, NY": [
        "IBM",
        "Google",
        "Microsoft",
        "Amazon",
        "Bloomberg",
        "JPMorgan Chase",
        "Goldman Sachs",
        "Datadog",
        "Spotify",
        "Morgan Stanley"
    ],

    "Australia, AS": [
        "Atlassian",
        "Canva",
        "Telstra",
        "Commonwealth Bank",
        "ANZ",
        "Westpac",
        "NAB",
        "WiseTech Global",
        "Macquarie Group",
        "REA Group"
    ],

    "Ahemdabad, IN": [
        "eInfochips",
        "TatvaSoft",
        "Radixweb",
        "Gateway Group",
        "Simform",
        "Zealous System",
        "Hidden Brains",
        "Cygnet Infotech",
        "OpenXcell",
        "Silver Touch Technologies"
    ],

    "Dubai, DU": [
        "Emirates",
        "Emirates NBD",
        "Careem",
        "Noon",
        "du",
        "Etisalat",
        "Mashreq",
        "Dubai Holding",
        "Property Finder",
        "Talabat"
    ],

    "Mumbai, IN": [
        "Tata Consultancy Services",
        "Reliance Industries",
        "Infosys",
        "Wipro",
        "HDFC Bank",
        "ICICI Bank",
        "Accenture",
        "Capgemini",
        "LTIMindtree",
        "Tech Mahindra"
    ],

    "London, UK": [
        "Google",
        "Microsoft",
        "Amazon",
        "Meta",
        "Barclays",
        "HSBC",
        "Lloyds Banking Group",
        "BT Group",
        "Revolut",
        "Monzo"
    ],

    "Bangalore, IN": [
        "Infosys",
        "Wipro",
        "Tata Consultancy Services",
        "Accenture",
        "Amazon",
        "Microsoft",
        "Google",
        "Flipkart",
        "Swiggy",
        "Razorpay"
    ],

    "Hyderabad, IN": [
        "Microsoft",
        "Google",
        "Amazon",
        "Deloitte",
        "Oracle",
        "Infosys",
        "Tata Consultancy Services",
        "Wipro",
        "ServiceNow",
        "Salesforce"
    ]
}
    
    locations = [
       "New York, NY","Australia, AS","Ahemdabad, IN","Dubai, DU","Mumbai, IN","London, UK","Bangalore, IN", "Hyderabad, IN"
    ]
    
    salary_by_location = {
    "Data Scientist": {
        "New York, NY": "$110,000 - $150,000 / yr",
        "Australia": "$75,000 - $105,000 / yr",
        "Ahmedabad, IN": "₹6,00,000 - ₹14,00,000 / yr",
        "Dubai, UAE": "AED 180,000 - AED 300,000 / yr",
        "Mumbai, IN": "₹8,00,000 - ₹18,00,000 / yr",
        "London, UK": "£60,000 - £90,000 / yr",
        "Bangalore, IN": "₹8,00,000 - ₹20,00,000 / yr",
        "Hyderabad, IN": "₹7,00,000 - ₹17,00,000 / yr"
    },

    "Data Analyst": {
        "New York, NY": "$75,000 - $105,000 / yr",
        "Australia": "$55,000 - $80,000 / yr",
        "Ahmedabad, IN": "₹4,00,000 - ₹9,00,000 / yr",
        "Dubai, UAE": "AED 120,000 - AED 210,000 / yr",
        "Mumbai, IN": "₹5,00,000 - ₹11,00,000 / yr",
        "London, UK": "£40,000 - £65,000 / yr",
        "Bangalore, IN": "₹5,00,000 - ₹12,00,000 / yr",
        "Hyderabad, IN": "₹4,50,000 - ₹10,00,000 / yr"
    },

    "Machine Learning Engineer": {
        "New York, NY": "$125,000 - $175,000 / yr",
        "Australia": "$85,000 - $125,000 / yr",
        "Ahmedabad, IN": "₹8,00,000 - ₹18,00,000 / yr",
        "Dubai, UAE": "AED 200,000 - AED 350,000 / yr",
        "Mumbai, IN": "₹10,00,000 - ₹22,00,000 / yr",
        "London, UK": "£65,000 - £100,000 / yr",
        "Bangalore, IN": "₹10,00,000 - ₹25,00,000 / yr",
        "Hyderabad, IN": "₹9,00,000 - ₹20,00,000 / yr"
    },

    "AI Engineer": {
        "New York, NY": "$130,000 - $180,000 / yr",
        "Australia": "$90,000 - $135,000 / yr",
        "Ahmedabad, IN": "₹8,00,000 - ₹20,00,000 / yr",
        "Dubai, UAE": "AED 220,000 - AED 380,000 / yr",
        "Mumbai, IN": "₹12,00,000 - ₹25,00,000 / yr",
        "London, UK": "£70,000 - £110,000 / yr",
        "Bangalore, IN": "₹12,00,000 - ₹28,00,000 / yr",
        "Hyderabad, IN": "₹10,00,000 - ₹22,00,000 / yr"
    },

    "Python Developer": {
        "New York, NY": "$90,000 - $125,000 / yr",
        "Australia": "$65,000 - $95,000 / yr",
        "Ahmedabad, IN": "₹5,00,000 - ₹12,00,000 / yr",
        "Dubai, UAE": "AED 140,000 - AED 240,000 / yr",
        "Mumbai, IN": "₹6,00,000 - ₹15,00,000 / yr",
        "London, UK": "£45,000 - £75,000 / yr",
        "Bangalore, IN": "₹7,00,000 - ₹16,00,000 / yr",
        "Hyderabad, IN": "₹5,00,000 - ₹12,00,000 / yr"
    },

    "Software Developer": {
        "New York, NY": "$95,000 - $135,000 / yr",
        "Australia": "$70,000 - $100,000 / yr",
        "Ahmedabad, IN": "₹5,00,000 - ₹12,00,000 / yr",
        "Dubai, UAE": "AED 150,000 - AED 260,000 / yr",
        "Mumbai, IN": "₹6,00,000 - ₹15,00,000 / yr",
        "London, UK": "£50,000 - £80,000 / yr",
        "Bangalore, IN": "₹7,00,000 - ₹18,00,000 / yr",
        "Hyderabad, IN": "₹6,00,000 - ₹14,00,000 / yr"
    },

    "Backend Developer": {
        "New York, NY": "$100,000 - $140,000 / yr",
        "Australia": "$75,000 - $105,000 / yr",
        "Ahmedabad, IN": "₹6,00,000 - ₹14,00,000 / yr",
        "Dubai, UAE": "AED 160,000 - AED 280,000 / yr",
        "Mumbai, IN": "₹7,00,000 - ₹17,00,000 / yr",
        "London, UK": "£50,000 - £85,000 / yr",
        "Bangalore, IN": "₹8,00,000 - ₹20,00,000 / yr",
        "Hyderabad, IN": "₹6,00,000 - ₹15,00,000 / yr"
    },

    "Frontend Developer": {
        "New York, NY": "$85,000 - $120,000 / yr",
        "Australia": "$60,000 - $90,000 / yr",
        "Ahmedabad, IN": "₹4,00,000 - ₹10,00,000 / yr",
        "Dubai, UAE": "AED 130,000 - AED 230,000 / yr",
        "Mumbai, IN": "₹5,00,000 - ₹13,00,000 / yr",
        "London, UK": "£40,000 - £70,000 / yr",
        "Bangalore, IN": "₹6,00,000 - ₹15,00,000 / yr",
        "Hyderabad, IN": "₹5,00,000 - ₹12,00,000 / yr"
    },

    "Full Stack Developer": {
        "New York, NY": "$105,000 - $145,000 / yr",
        "Australia": "$75,000 - $110,000 / yr",
        "Ahmedabad, IN": "₹6,00,000 - ₹15,00,000 / yr",
        "Dubai, UAE": "AED 170,000 - AED 300,000 / yr",
        "Mumbai, IN": "₹8,00,000 - ₹19,00,000 / yr",
        "London, UK": "£55,000 - £90,000 / yr",
        "Bangalore, IN": "₹8,00,000 - ₹22,00,000 / yr",
        "Hyderabad, IN": "₹7,00,000 - ₹18,00,000 / yr"
    },

    "Business Analyst": {
        "New York, NY": "$80,000 - $110,000 / yr",
        "Australia": "$60,000 - $85,000 / yr",
        "Ahmedabad, IN": "₹4,00,000 - ₹9,00,000 / yr",
        "Dubai, UAE": "AED 130,000 - AED 220,000 / yr",
        "Mumbai, IN": "₹5,00,000 - ₹12,00,000 / yr",
        "London, UK": "£42,000 - £68,000 / yr",
        "Bangalore, IN": "₹6,00,000 - ₹14,00,000 / yr",
        "Hyderabad, IN": "₹5,00,000 - ₹11,00,000 / yr"
    }
}
    
    exp_levels = ["0-1 years (Fresher)", "1-2 years", "2-3 years", "3-5 years", "Fresher / Entry Level"]
    
    # Set seed for reproducibility
    random.seed(42)
    
    records = []
    job_counter = 1001
    
    # Generate exactly 120 realistic records (12 per role)
    selected_location = st.selectbox(
    "📍 Select Location",
    locations)

    for role_info in salary_by_location:
        role = role_info["role"]

        role_info["location"] = selected_location
        role_info["salary"] = salary_by_location[role][selected_location]
        for i in range(12):
            job_id = f"JOB{job_counter}"
            job_counter += 1
            
            company = random.choice(companies_by_location[selected_location])
            location = random.choice(locations)
            
            # Pick a subset of skills
            num_skills = random.randint(6, len(role_info["skills"]))
            selected_skills = random.sample(role_info["skills"], num_skills)
            skills_str = ", ".join(selected_skills)
            
            education = random.choice(role_info["education"])
            exp = random.choice(exp_levels)
            
            desc = f"{role_info['desc']} Seeking candidate skilled in {selected_skills[0]} and {selected_skills[1]}. Ideal for candidates aiming to work with modern engineering standards."
            
            records.append({
            "job_id": job_id,
            "job_title": role,
            "company": company,
            "location": location,
            "skills": skills_str,
            "experience": exp,
            "education": education,
            "job_description": desc,
            "salary": salary_by_location[role][location]
            })
            
    # Write CSV files
    fieldnames = ["job_id", "job_title", "company", "location", "skills", "experience", "education", "job_description", "salary"]
    
    for path in file_paths:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
            
    print(f"Successfully generated dataset with {len(records)} records.")

if __name__ == "__main__":
    generate_jobs_dataset()
