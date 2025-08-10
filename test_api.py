import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_stats():
    """Test the stats endpoint"""
    response = requests.get(f"{BASE_URL}/stats")
    print("Stats:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_match_job():
    """Test the job matching endpoint"""
    job_data = {
        "title": "Machine Learning Engineer",
        "description": "We are looking for a Machine Learning Engineer with experience in Python, TensorFlow, and deep learning. The ideal candidate should have experience with data preprocessing, model training, and deployment.",
        "skills": ["Python", "TensorFlow", "Machine Learning", "Deep Learning", "Data Preprocessing"],
        "top_k": 10
    }
    
    response = requests.post(f"{BASE_URL}/match", json=job_data)
    print("Job Matching Results:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_add_resume():
    """Test adding a new resume"""
    resume_data = {
        "career_objective": "Experienced data scientist with expertise in machine learning and statistical analysis. Passionate about solving complex business problems using data-driven approaches.",
        "skills": ["Python", "Machine Learning", "Statistical Analysis", "SQL", "Data Visualization"],
        "educational_institution_name": ["Stanford University"],
        "degree_names": ["Master of Science in Statistics"],
        "passing_years": ["2020"],
        "educational_results": ["3.8 GPA"],
        "result_types": ["GPA"],
        "major_field_of_studies": ["Statistics"],
        "professional_company_names": ["Google", "Microsoft"]
    }
    
    response = requests.post(f"{BASE_URL}/resume", json=resume_data)
    print("Add Resume Result:")
    print(json.dumps(response.json(), indent=2))
    print()

def test_rebuild_index():
    """Test rebuilding the index"""
    response = requests.post(f"{BASE_URL}/index")
    print("Rebuild Index Result:")
    print(json.dumps(response.json(), indent=2))
    print()

if __name__ == "__main__":
    print("Testing TalentAI API")
    print("=" * 50)
    
    # Test health endpoint
    test_health()
    
    # Test stats endpoint
    test_stats()
    
    # Test job matching
    test_match_job()
    
    # Test adding a resume
    test_add_resume()
    
    # Test rebuilding index (optional - can take time)
    # test_rebuild_index()
