from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import json
import os
import ast
import unicodedata
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables to store the model and data
model = None
index = None
metadata = []
resumes_df = None
esco_df = None
onet_df = None

# Configuration
STORE_DIR = "store"
RESUMES_FILE = os.path.join(STORE_DIR, "resumes_filtered_1000.csv")
INDEX_FILE = os.path.join(STORE_DIR, "index.faiss")
METADATA_FILE = os.path.join(STORE_DIR, "metadata.jsonl")
ESCO_FILE = os.path.join(STORE_DIR, "skills_en.csv")
ONET_FILE = os.path.join(STORE_DIR, "Skills_ONET.txt")

def load_model():
    """Load the sentence transformer model"""
    global model
    if model is None:
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    return model

def load_data():
    """Load resumes data and metadata"""
    global resumes_df, metadata
    
    if resumes_df is None:
        logger.info("Loading resumes data...")
        resumes_df = pd.read_csv(RESUMES_FILE)
    
    if not metadata:
        logger.info("Loading metadata...")
        metadata.clear()
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                metadata.append(json.loads(line.strip()))
    
    return resumes_df, metadata

def load_index():
    """Load FAISS index"""
    global index
    if index is None:
        logger.info("Loading FAISS index...")
        index = faiss.read_index(INDEX_FILE)
    return index

def load_skill_databases():
    """Load ESCO and O*NET skill databases"""
    global esco_df, onet_df
    
    if esco_df is None:
        logger.info("Loading ESCO skills database...")
        esco_df = pd.read_csv(ESCO_FILE)
        esco_df['preferredLabel_norm'] = esco_df['preferredLabel'].apply(normalize_term)
    
    if onet_df is None:
        logger.info("Loading O*NET skills database...")
        onet_df = pd.read_csv(ONET_FILE, sep="\t")  # Tab-separated
        onet_df['Element Name norm'] = onet_df['Element Name'].apply(normalize_term)
    
    return esco_df, onet_df

def normalize_term(s: str) -> str:
    """Normalize a term using NFKC normalization, lowercase, and whitespace normalization"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = " ".join(s.split())
    return s

def parse_skills_field(raw):
    """Parse skills field from various formats (list, string, etc.)"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    s = str(raw).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val if str(x).strip()]
        except Exception:
            pass
    for sep in [";", "|", ","]:
        if sep in s:
            return [t for t in (part.strip() for part in s.split(sep)) if t]
    return [s] if s else []

def normalize_skills(raw_skills):
    """
    Normalizes skills using local ESCO and O*NET files.
    
    Args:
        raw_skills (str): Raw skills from a resume.
        
    Returns:
        tuple: (list of ESCO IDs, dict of O*NET skills with importance)
    """
    # Load skill databases
    esco_df, onet_df = load_skill_databases()
    
    # Parse skills using the improved parser
    skills_list = parse_skills_field(raw_skills)
    
    # Normalize skills using the same normalization function
    skills_list_norm = [normalize_term(s) for s in skills_list]
    
    if not skills_list_norm:
        return [], {}
    
    # Match ESCO skills using normalized labels
    mask = esco_df["preferredLabel_norm"].isin(skills_list_norm)
    esco_ids = esco_df.loc[mask, "conceptUri"].tolist()
    
    # Build fast O*NET lookup for importance values
    onet_importance = (
        onet_df.groupby("Element Name norm")["Data Value"]
        .max()
        .to_dict()
    )
    
    # O*NET match using normalized names
    onet_skills = {}
    for skill in skills_list_norm:
        val = onet_importance.get(skill)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            onet_skills[skill] = float(val)
    
    return esco_ids, onet_skills

def create_resume_embedding(resume_text):
    """Create embedding for resume text"""
    model = load_model()
    embedding = model.encode([resume_text], convert_to_tensor=False)
    return embedding[0]

def match_job(job_embedding, job_skills, top_k=10, filters=None):
    """
    Match a job to candidates using the FAISS index
    """
    index = load_index()
    _, metadata = load_data()
    
    logger.info(f"Index size: {index.ntotal}, Metadata size: {len(metadata)}")
    logger.info(f"Job skills: {job_skills}")
    logger.info(f"Filters: {filters}")
    
    # Normalize job embedding for cosine similarity
    job_embedding_norm = job_embedding / np.linalg.norm(job_embedding)
    
    # Precompute job normalization ONCE
    job_esco_ids_list, job_onet_skills = normalize_skills(job_skills)
    job_esco_ids = set(job_esco_ids_list)
    job_skills_norm = [normalize_term(s) for s in job_skills]
    
    # Search the index
    distances, indices = index.search(np.array([job_embedding_norm]), top_k)
    logger.info(f"FAISS search returned {len(indices[0])} results")
    
    candidates = []
    for i, sim in zip(indices[0], distances[0]):
        if i == -1 or i >= len(metadata):
            logger.warning(f"Skipping invalid index {i}")
            continue
            
        candidate = metadata[i]
        
        # Apply filters if any
        if filters:
            filter_match = True
            for k, v in filters.items():
                candidate_value = candidate.get(k)
                
                # Handle major_field_of_studies filter specifically
                if k == 'major_field_of_studies':
                    # Parse the string representation of the list
                    if isinstance(candidate_value, str):
                        try:
                            # Remove brackets and split by comma, then clean up
                            if candidate_value.startswith('[') and candidate_value.endswith(']'):
                                candidate_majors = [s.strip().strip("'\"") for s in candidate_value[1:-1].split(',') if s.strip()]
                            else:
                                candidate_majors = [candidate_value.strip()]
                        except:
                            candidate_majors = []
                    else:
                        candidate_majors = candidate_value if isinstance(candidate_value, list) else []
                    
                    # Check if any of the required majors match any of the candidate's majors
                    if not any(required_major.lower() in [cm.lower() for cm in candidate_majors] for required_major in v):
                        filter_match = False
                        break
                else:
                    # For other filters, use direct comparison
                    if k not in candidate or candidate.get(k) != v:
                        filter_match = False
                        break
            if not filter_match:
                continue
        
        # Calculate semantic score (cosine similarity)
        semantic_score = sim
        
        # Calculate skill overlap using ESCO IDs
        candidate_esco_ids = set(candidate.get('esco_ids', []))
        skill_overlap = len(candidate_esco_ids & job_esco_ids)
        
        # Calculate ONET skill weight using normalized skill names
        candidate_onet_skills = candidate.get('onet_skills', {})
        # Use overlap of O*NET names, weighted by candidate importance
        onet_weight = sum(
            candidate_onet_skills.get(skill_name, 0) 
            for skill_name in job_onet_skills.keys()
        )
        
        # Calculate final score
        final_score = 1 * semantic_score + 0.2 * skill_overlap + 0.1 * onet_weight
        
        explanation = "Semantic match"
        if skill_overlap > 0:
            explanation += f", skill overlap: {skill_overlap}"
        if onet_weight > 0:
            explanation += f", ONET skill weight: {onet_weight:.2f}"
        
        candidates.append({
            'candidate_id': int(i),
            'score': float(final_score),
            'explanation': explanation,
            'metadata': candidate
        })
    
    logger.info(f"Returning {len(candidates)} candidates")
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[:top_k]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "TalentAI API is running"})

@app.route('/test-match', methods=['POST'])
def test_match():
    """Simple test endpoint without filters"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        title = data.get('title', '')
        description = data.get('description', '')
        skills = data.get('skills', [])
        top_k = data.get('top_k', 5)
        
        # Create job text for embedding
        job_text = f"{title} {description} {' '.join(skills)}"
        
        # Create job embedding
        model = load_model()
        job_embedding = create_resume_embedding(job_text)
        
        # Match job to candidates without filters
        candidates = match_job(job_embedding, skills, top_k, None)
        
        # Get full candidate information
        resumes_df, _ = load_data()
        results = []
        
        for candidate in candidates:
            candidate_id = candidate['candidate_id']
            if candidate_id < len(resumes_df):
                candidate_info = resumes_df.iloc[candidate_id].to_dict()
                
                # Helper function to handle NaN values
                def clean_value(value):
                    if pd.isna(value):
                        return None
                    return value
                
                result = {
                    'candidate_id': candidate_id,
                    'score': candidate['score'],
                    'explanation': candidate['explanation'],
                    'career_objective': clean_value(candidate_info.get('career_objective')),
                    'skills': clean_value(candidate_info.get('skills')),
                    'educational_institution_name': clean_value(candidate_info.get('educational_institution_name')),
                    'degree_names': clean_value(candidate_info.get('degree_names')),
                    'passing_years': clean_value(candidate_info.get('passing_years')),
                    'major_field_of_studies': clean_value(candidate_info.get('major_field_of_studies')),
                    'professional_company_names': clean_value(candidate_info.get('professional_company_names'))
                }
                results.append(result)
        
        return jsonify({
            "status": "success",
            "job_title": title,
            "candidates_found": len(results),
            "candidates": results
        })
        
    except Exception as e:
        logger.error(f"Error in test match: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/index', methods=['POST'])
def rebuild_index():
    """
    Rebuild the FAISS index and metadata from the resumes data
    """
    try:
        logger.info("Starting index rebuild...")
        
        # Load data
        resumes_df, _ = load_data()
        model = load_model()
        
        # Create embeddings for all resumes
        logger.info("Creating embeddings...")
        resume_texts = []
        for _, row in resumes_df.iterrows():
            career_obj = str(row.get('career_objective', ''))
            skills = str(row.get('skills', ''))
            resume_text = f"{career_obj} {skills}"
            resume_texts.append(resume_text)
        
        embeddings = model.encode(resume_texts, convert_to_tensor=False)
        embeddings = np.array(embeddings)
        
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create new FAISS index
        logger.info("Building FAISS index...")
        dimension = embeddings_norm.shape[1]
        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(embeddings_norm)
        
        # Create metadata
        logger.info("Creating metadata...")
        new_metadata = []
        for _, row in resumes_df.iterrows():
            esco_ids, onet_skills = normalize_skills(row.get('skills'))
            
            metadata_entry = {
                'educational_institution_name': row.get('educational_institution_name', []),
                'passing_years': row.get('passing_years', []),
                'educational_results': row.get('educational_results', []),
                'result_types': row.get('result_types', []),
                'major_field_of_studies': row.get('major_field_of_studies', []),
                'professional_company_names': row.get('professional_company_names', []),
                'esco_ids': esco_ids,
                'onet_skills': onet_skills
            }
            new_metadata.append(metadata_entry)
        
        # Save new index and metadata
        logger.info("Saving new index and metadata...")
        faiss.write_index(new_index, INDEX_FILE)
        
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            for entry in new_metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Update global variables
        global index, metadata
        index = new_index
        metadata = new_metadata
        
        logger.info("Index rebuild completed successfully")
        return jsonify({
            "status": "success",
            "message": "Index rebuilt successfully",
            "resume_count": len(resumes_df),
            "embedding_dimension": dimension
        })
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/match', methods=['POST'])
def match_job_endpoint():
    """
    Match a job to candidates
    Expected JSON:
    {
        "title": "Job Title",
        "description": "Job Description",
        "skills": ["skill1", "skill2", ...],
        "top_k": 10,
        "filters": {"field": "value"}
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        title = data.get('title', '')
        description = data.get('description', '')
        skills = data.get('skills', [])
        top_k = data.get('top_k', 10)
        filters = data.get('filters', None)
        
        # Create job text for embedding
        job_text = f"{title} {description} {' '.join(skills)}"
        
        # Create job embedding
        model = load_model()
        job_embedding = create_resume_embedding(job_text)
        
        # Match job to candidates (temporarily disable filters for debugging)
        candidates = match_job(job_embedding, skills, top_k, None)  # filters
        logger.info(f"Found {len(candidates)} candidates from match_job function")
        
        # Get full candidate information
        resumes_df, _ = load_data()
        logger.info(f"Loaded {len(resumes_df)} resumes from CSV")
        results = []
        
        for candidate in candidates:
            candidate_id = candidate['candidate_id']
            if candidate_id < len(resumes_df):
                candidate_info = resumes_df.iloc[candidate_id].to_dict()
                
                # Helper function to handle NaN values
                def clean_value(value):
                    if pd.isna(value):
                        return None
                    return value
                
                result = {
                    'candidate_id': candidate_id,
                    'score': candidate['score'],
                    'explanation': candidate['explanation'],
                    'career_objective': clean_value(candidate_info.get('career_objective')),
                    'skills': clean_value(candidate_info.get('skills')),
                    'educational_institution_name': clean_value(candidate_info.get('educational_institution_name')),
                    'degree_names': clean_value(candidate_info.get('degree_names')),
                    'passing_years': clean_value(candidate_info.get('passing_years')),
                    'major_field_of_studies': clean_value(candidate_info.get('major_field_of_studies')),
                    'professional_company_names': clean_value(candidate_info.get('professional_company_names'))
                }
                results.append(result)
            else:
                logger.warning(f"Candidate ID {candidate_id} is out of range (max: {len(resumes_df)})")
        
        return jsonify({
            "status": "success",
            "job_title": title,
            "candidates_found": len(results),
            "candidates": results
        })
        
    except Exception as e:
        logger.error(f"Error matching job: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/resume', methods=['POST'])
def add_resume():
    """
    Add a new resume to the index
    Expected JSON:
    {
        "career_objective": "Career objective text",
        "skills": ["skill1", "skill2", ...],
        "educational_institution_name": ["institution1", "institution2"],
        "degree_names": ["degree1", "degree2"],
        "passing_years": ["year1", "year2"],
        "educational_results": ["result1", "result2"],
        "result_types": ["type1", "type2"],
        "major_field_of_studies": ["field1", "field2"],
        "professional_company_names": ["company1", "company2"]
    }
    """
    try:
        # Update global variables
        global resumes_df
        
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        # Load current data
        resumes_df, metadata = load_data()
        index = load_index()
        model = load_model()
        
        # Create resume text for embedding
        career_obj = data.get('career_objective', '')
        skills = data.get('skills', [])
        skills_text = ', '.join(skills) if isinstance(skills, list) else str(skills)
        resume_text = f"{career_obj} {skills_text}"
        
        # Create embedding
        new_embedding = create_resume_embedding(resume_text)
        new_embedding_norm = new_embedding / np.linalg.norm(new_embedding)
        
        # Add to FAISS index
        index.add(np.array([new_embedding_norm]))
        
        # Create metadata entry
        esco_ids, onet_skills = normalize_skills(skills)
        new_metadata_entry = {
            'educational_institution_name': data.get('educational_institution_name', []),
            'passing_years': data.get('passing_years', []),
            'educational_results': data.get('educational_results', []),
            'result_types': data.get('result_types', []),
            'major_field_of_studies': data.get('major_field_of_studies', []),
            'professional_company_names': data.get('professional_company_names', []),
            'esco_ids': esco_ids,
            'onet_skills': onet_skills
        }
        
        # Add to metadata list
        metadata.append(new_metadata_entry)
        
        # Add to resumes DataFrame
        new_resume_row = {
            'career_objective': career_obj,
            'skills': skills_text,
            'educational_institution_name': data.get('educational_institution_name', []),
            'degree_names': data.get('degree_names', []),
            'passing_years': data.get('passing_years', []),
            'educational_results': data.get('educational_results', []),
            'result_types': data.get('result_types', []),
            'major_field_of_studies': data.get('major_field_of_studies', []),
            'professional_company_names': data.get('professional_company_names', [])
        }
        
        new_resume_df = pd.DataFrame([new_resume_row])
        resumes_df = pd.concat([resumes_df, new_resume_df], ignore_index=True)
        
        # Save updated data
        faiss.write_index(index, INDEX_FILE)
        
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        resumes_df.to_csv(RESUMES_FILE, index=False)
        
        new_resume_id = len(resumes_df) - 1
        
        logger.info(f"Added new resume with ID: {new_resume_id}")
        
        return jsonify({
            "status": "success",
            "message": "Resume added successfully",
            "resume_id": new_resume_id,
            "total_resumes": len(resumes_df)
        })
        
    except Exception as e:
        logger.error(f"Error adding resume: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the current index"""
    try:
        resumes_df, metadata = load_data()
        index = load_index()
        
        return jsonify({
            "status": "success",
            "total_resumes": len(resumes_df),
            "index_size": index.ntotal,
            "embedding_dimension": index.d,
            "metadata_entries": len(metadata)
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Pre-load data on startup
    try:
        load_model()
        load_data()
        load_index()
        load_skill_databases()
        logger.info("All data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading data on startup: {str(e)}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
