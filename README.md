# TalentAI API

A Flask-based API for intelligent resume-job matching using semantic search and FAISS indexing.

## Features

- **Semantic Resume-Job Matching**: Uses sentence transformers for semantic similarity
- **FAISS Indexing**: Fast similarity search using FAISS
- **Dynamic Resume Addition**: Add new resumes to the index in real-time
- **Flexible Filtering**: Filter candidates based on various criteria
- **RESTful API**: Clean REST endpoints for easy integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required data files in the `store/` directory:
   - `resumes_filtered_1000.csv` - Resume dataset
   - `index.faiss` - FAISS index file
   - `metadata.jsonl` - Resume metadata

## Running the API

```bash
python app.py
```

The API will start on `http://localhost:5000`

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy",
  "message": "TalentAI API is running"
}
```

### 2. Get Statistics
**GET** `/stats`

Get statistics about the current index.

**Response:**
```json
{
  "status": "success",
  "total_resumes": 1000,
  "index_size": 1000,
  "embedding_dimension": 768,
  "metadata_entries": 1000
}
```

### 3. Rebuild Index
**POST** `/index`

Rebuild the FAISS index and metadata from the resumes data. This is useful when you want to update the entire index.

**Response:**
```json
{
  "status": "success",
  "message": "Index rebuilt successfully",
  "resume_count": 1000,
  "embedding_dimension": 768
}
```

### 4. Match Job to Candidates
**POST** `/match`

Find the best candidates for a given job posting.

**Request Body:**
```json
{
  "title": "Machine Learning Engineer",
  "description": "We are looking for a Machine Learning Engineer with experience in Python, TensorFlow, and deep learning.",
  "skills": ["Python", "TensorFlow", "Machine Learning", "Deep Learning"],
  "top_k": 10,
  "filters": {
    "major_field_of_studies": ["Computer Science"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "job_title": "Machine Learning Engineer",
  "candidates_found": 5,
  "candidates": [
    {
      "candidate_id": 123,
      "score": 0.85,
      "explanation": "Semantic match, skill overlap: 3",
      "career_objective": "Experienced ML engineer...",
      "skills": "['Python', 'Machine Learning', 'TensorFlow']",
      "educational_institution_name": "['MIT']",
      "degree_names": "['Master of Science']",
      "passing_years": "['2020']",
      "major_field_of_studies": "['Computer Science']",
      "professional_company_names": "['Google']"
    }
  ]
}
```

### 5. Add New Resume
**POST** `/resume`

Add a new resume to the index.

**Request Body:**
```json
{
  "career_objective": "Experienced data scientist with expertise in machine learning and statistical analysis.",
  "skills": ["Python", "Machine Learning", "Statistical Analysis", "SQL"],
  "educational_institution_name": ["Stanford University"],
  "degree_names": ["Master of Science in Statistics"],
  "passing_years": ["2020"],
  "educational_results": ["3.8 GPA"],
  "result_types": ["GPA"],
  "major_field_of_studies": ["Statistics"],
  "professional_company_names": ["Google", "Microsoft"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Resume added successfully",
  "resume_id": 1001,
  "total_resumes": 1001
}
```

## Usage Examples

### Python Client Example

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:5000"

# Match a job to candidates
job_data = {
    "title": "Data Scientist",
    "description": "Looking for a data scientist with Python and machine learning experience",
    "skills": ["Python", "Machine Learning", "SQL", "Statistics"],
    "top_k": 5
}

response = requests.post(f"{BASE_URL}/match", json=job_data)
candidates = response.json()["candidates"]

for candidate in candidates:
    print(f"Candidate {candidate['candidate_id']}: Score {candidate['score']:.3f}")
    print(f"Skills: {candidate['skills']}")
    print("---")
```

### cURL Examples

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Get Stats:**
```bash
curl http://localhost:5000/stats
```

**Match Job:**
```bash
curl -X POST http://localhost:5000/match \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Machine Learning Engineer",
    "description": "Looking for ML engineer with Python experience",
    "skills": ["Python", "Machine Learning"],
    "top_k": 5
  }'
```

**Add Resume:**
```bash
curl -X POST http://localhost:5000/resume \
  -H "Content-Type: application/json" \
  -d '{
    "career_objective": "Experienced data scientist",
    "skills": ["Python", "Machine Learning"],
    "educational_institution_name": ["MIT"],
    "degree_names": ["Master of Science"],
    "passing_years": ["2020"],
    "educational_results": ["3.8 GPA"],
    "result_types": ["GPA"],
    "major_field_of_studies": ["Computer Science"],
    "professional_company_names": ["Google"]
  }'
```

## Testing

Run the test script to verify the API functionality:

```bash
python test_api.py
```

## Architecture

The API uses the following components:

1. **Sentence Transformers**: For creating semantic embeddings of resumes and job descriptions
2. **FAISS**: For fast similarity search using cosine similarity
3. **Pandas**: For data manipulation and storage
4. **Flask**: Web framework for the REST API

## Scoring Algorithm

The matching score is calculated using:
- **70% Semantic Similarity**: Cosine similarity between job and resume embeddings
- **20% Skill Overlap**: Number of matching skills between job and resume
- **10% ONET Skill Weight**: Weighted importance of skills based on O*NET database

## Data Structure

The API expects the following data files:

- `resumes_filtered_1000.csv`: CSV file with resume data
- `index.faiss`: FAISS index file for fast similarity search
- `metadata.jsonl`: JSONL file with resume metadata

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `500`: Internal Server Error

Error responses include a descriptive message:
```json
{
  "status": "error",
  "message": "Error description"
}
```

## Performance

- **Index Loading**: ~2-3 seconds on startup
- **Job Matching**: ~100-200ms per query
- **Resume Addition**: ~500ms per resume
- **Index Rebuild**: ~30-60 seconds for 1000 resumes

## Future Enhancements

- ESCO/O*NET skill normalization
- Advanced filtering options
- Batch operations
- Authentication and rate limiting
- Caching for improved performance
