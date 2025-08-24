import os
import re
import json
import hashlib
import logging
from typing import TypedDict, List, Dict, Any, Optional

import faiss
import numpy as np
import yaml  
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from langgraph.graph import StateGraph, END

from app.db.database import get_db
from app.models.jobs import Job, Proposal, JobRelevance
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEAM_PROFILES_MD_PATH = os.path.join(BASE_DIR, "agents", "team_profiles.md")
PROJECTS_MD_PATH = os.path.join(BASE_DIR, "agents", "projects.md")
PROPOSAL_TEMPLATE_MD_PATH = os.path.join(BASE_DIR, "agents", "proposal_template.md")

RAG_DATA_DIR = os.path.join(BASE_DIR, "rag_data")
FAISS_INDEX_FILE_NAME = "vector_store.faiss"
FAISS_METADATA_FILE_NAME = "metadata.json"
FILE_HASHES_FILE_NAME = "file_hashes.json"

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_GENERATION_MODEL = "gpt-5-2025-08-07"
EMBEDDING_DIM = 1536
NUM_RETRIEVED_CHUNKS = 8 

router = APIRouter()

def get_openai_client() -> OpenAI:
    """Initialize and return an OpenAI client."""
    api_key = os.getenv("OPEN_AI_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables (OPEN_AI_KEY).")
    return OpenAI(api_key=api_key)

def load_proposal_template(file_path: str) -> str:
    """Load content from the proposal template markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Proposal template file not found: {file_path}")
        return """Hi {{client_name}},\n\n{{introduction}}\n\nWhat I Can Offer for Your Project:\n\nAfter reviewing your project description, I believe I’m well-equipped to provide exactly what you’re looking for. Here’s how I can help:\n{{offer_section}}\n\n{{skills}}\n\n{{projects}}\n\n{{next_steps}}"""
    except Exception as e:
        logger.error(f"Error reading proposal template file {file_path}: {str(e)}")
        return ""

class FAISSIndexManager:
    """
    Manages the creation, loading, and querying of a unified FAISS index
    from structured markdown files (profiles and projects).
    """
    def __init__(self, openai_client: OpenAI,
                 profiles_md_path: str,
                 projects_md_path: str,
                 rag_data_dir: str,
                 index_file_name: str,
                 metadata_file_name: str,
                 hashes_file_name: str):
        self.openai_client = openai_client
        self.profiles_md_path = profiles_md_path
        self.projects_md_path = projects_md_path

        os.makedirs(rag_data_dir, exist_ok=True)
        self.index_path = os.path.join(rag_data_dir, index_file_name)
        self.metadata_path = os.path.join(rag_data_dir, metadata_file_name)
        self.hashes_path = os.path.join(rag_data_dir, hashes_file_name)

        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[Dict[str, Any]] = []

        self._load_or_build_index()

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        if not os.path.exists(file_path):
            return None
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _load_stored_hashes(self) -> Dict[str, Optional[str]]:
        if not os.path.exists(self.hashes_path):
            return {}
        try:
            with open(self.hashes_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding {self.hashes_path}. Will rebuild index.")
            return {}

    def _save_current_hashes(self, current_hashes: Dict[str, Optional[str]]):
        with open(self.hashes_path, 'w') as f:
            json.dump(current_hashes, f)

    def _check_if_rebuild_needed(self) -> bool:
        if not all(os.path.exists(p) for p in [self.index_path, self.metadata_path, self.hashes_path]):
            logger.info("Index, metadata, or hashes file not found. Rebuilding index.")
            return True

        stored_hashes = self._load_stored_hashes()
        current_hashes = {
            "profiles": self._get_file_hash(self.profiles_md_path),
            "projects": self._get_file_hash(self.projects_md_path),
        }

        if stored_hashes.get("profiles") != current_hashes["profiles"] or \
           stored_hashes.get("projects") != current_hashes["projects"]:
            logger.info("A markdown data file has changed. Rebuilding index.")
            return True

        logger.info("Data files are unchanged. No rebuild needed.")
        return False

    def _parse_and_chunk_files(self) -> List[Dict[str, Any]]:
        """
        <<< NEW & ROBUST: Parses YAML front matter from markdown files.
        This replaces the old, brittle regex and text splitting logic.
        """
        all_chunks = []
        
        def process_file(file_path: str, doc_type: str):
            logger.info(f"Parsing {doc_type} data from: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                logger.warning(f"{doc_type.capitalize()} file not found at {file_path}. Skipping.")
                return

            documents = re.split(r'\n---\n', content.strip())
            for i, doc_content in enumerate(documents):
                if not doc_content.strip():
                    continue
                try:
                    # The first part of the document is YAML front matter
                    front_matter = next(yaml.safe_load_all(doc_content))
                    
                    # Create a rich text block for semantic embedding
                    name = front_matter.get('name', f"Unnamed {doc_type}")
                    description = front_matter.get('description', 'No description provided.')
                    text_for_embedding = f"{doc_type.capitalize()}: {name}. Description: {description}"
                    
                    # Store ALL parsed data in metadata for precise context generation
                    chunk_metadata = {
                        "id": f"{doc_type}_{i}",
                        "doc_type": doc_type,
                        "text_for_embedding": text_for_embedding,
                        **front_matter
                    }
                    all_chunks.append(chunk_metadata)
                    logger.debug(f"Created chunk for '{name}'. ID: {chunk_metadata['id']}")

                except Exception as e:
                    logger.error(f"Error parsing document #{i+1} in {file_path}: {e}")

        process_file(self.profiles_md_path, "profile")
        process_file(self.projects_md_path, "project")
        
        return all_chunks

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            response = self.openai_client.embeddings.create(input=texts, model=OPENAI_EMBEDDING_MODEL)
            return np.array([item.embedding for item in response.data]).astype('float32')
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI: {e}", exc_info=True)
            raise

    def _build_index(self):
        logger.info("Building new unified FAISS index from structured data...")
        self.chunks_metadata = self._parse_and_chunk_files()
        
        if not self.chunks_metadata:
            logger.error("No chunks were created from source files. Aborting index build.")
            self.index = None
            return

        texts_for_embedding = [chunk["text_for_embedding"] for chunk in self.chunks_metadata]

        try:
            embeddings_np = self._get_embeddings(texts_for_embedding)
            self.index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.index.add(embeddings_np)

            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.chunks_metadata, f, indent=2)
            
            num_profiles = len([c for c in self.chunks_metadata if c['doc_type'] == 'profile'])
            num_projects = len([c for c in self.chunks_metadata if c['doc_type'] == 'project'])
            logger.info(f"FAISS index built and saved successfully with {self.index.ntotal} total chunks ({num_profiles} profiles, {num_projects} projects).")

        except Exception as e:
            logger.error(f"Failed to build and save index: {e}", exc_info=True)
            self.index = None
            return

        current_hashes = {
            "profiles": self._get_file_hash(self.profiles_md_path),
            "projects": self._get_file_hash(self.projects_md_path),
        }
        self._save_current_hashes(current_hashes)

    def _load_index(self) -> bool:
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'r') as f:
                self.chunks_metadata = json.load(f)
            logger.info(f"Unified FAISS index and metadata loaded successfully with {len(self.chunks_metadata)} chunks.")
            return True
        except Exception as e:
            logger.error(f"Failed to load FAISS index or metadata: {e}", exc_info=True)
            return False

    def _load_or_build_index(self):
        if self._check_if_rebuild_needed():
            self._build_index()
        elif not self._load_index():
            logger.warning("Failed to load existing index. Attempting rebuild.")
            self._build_index()

        if self.index is None:
            logger.critical("CRITICAL: FAISS index is not available. RAG queries will fail.")

    def query(self, query_text: str, k: int) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is not initialized or empty. Cannot perform query.")
            return []
        
        try:
            query_embedding = self._get_embeddings([query_text])
            distances, indices = self.index.search(query_embedding, k=min(k, self.index.ntotal))

            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.chunks_metadata):
                    result_metadata = self.chunks_metadata[idx].copy()
                    result_metadata["distance"] = float(distances[0][i])
                    results.append(result_metadata)
            
            logger.info(f"RAG query returned {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS query: {e}", exc_info=True)
            return []

GLOBAL_FAISS_MANAGER: Optional[FAISSIndexManager] = None
try:
    logger.info("Initializing FAISSIndexManager with structured data sources...")
    _openai_client = get_openai_client()
    GLOBAL_FAISS_MANAGER = FAISSIndexManager(
        openai_client=_openai_client,
        profiles_md_path=TEAM_PROFILES_MD_PATH,
        projects_md_path=PROJECTS_MD_PATH,
        rag_data_dir=RAG_DATA_DIR,
        index_file_name=FAISS_INDEX_FILE_NAME,
        metadata_file_name=FAISS_METADATA_FILE_NAME,
        hashes_file_name=FILE_HASHES_FILE_NAME
    )
except ValueError as ve:
    logger.critical(f"CRITICAL: {ve}")
    GLOBAL_FAISS_MANAGER = None
except Exception as e:
    logger.error(f"Failed to initialize FAISSIndexManager globally: {e}", exc_info=True)
    GLOBAL_FAISS_MANAGER = None


class ProposalState(TypedDict):
    job_id: str
    client_name: str
    job_title: str
    job_description: str
    final_proposal: str
    retrieved_context: str
    proposal_template: str
    db: Session
    relevance_score: Optional[float] 
    closest_profile_name: Optional[str] 

def execute_openai_call(prompt: str) -> str:
    """Executes the Chat Completion call to OpenAI."""
    try:
        completion = _openai_client.chat.completions.create(
            model=OPENAI_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2048, 
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"OpenAI call failed: {e}", exc_info=True)
        return "Error: Could not generate proposal content."

def get_job_details(state: ProposalState) -> ProposalState:
    db = state['db']
    job_id = state['job_id']
    logger.info(f"Fetching job details for job_id: {job_id}")
    job = db.query(Job).options(joinedload(Job.relevance)).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with id {job_id} not found")

    state['client_name'] = "Client" 
    state['job_title'] = job.title
    state['job_description'] = job.description
    state['relevance_score'] = job.relevance.score if job.relevance else None
    state['closest_profile_name'] = job.relevance.closest_profile_name if job.relevance else None
    logger.info(f"Job details loaded for '{job.title}'.")
    return state

def load_template_node(state: ProposalState) -> ProposalState:
    state['proposal_template'] = load_proposal_template(PROPOSAL_TEMPLATE_MD_PATH)
    logger.info("Proposal template loaded into state.")
    return state

def retrieve_context(state: ProposalState) -> ProposalState:
    """
    <<< REWRITTEN & ENHANCED: Performs a single, unified search and builds a
    clean, structured context string using the rich metadata from the index.
    """
    if GLOBAL_FAISS_MANAGER is None:
        logger.error("RAG system not initialized. Cannot retrieve context.")
        state['retrieved_context'] = "Error: RAG system is offline."
        return state

    query_text = f"Job Title: {state['job_title']}. Description: {state['job_description']}"
    logger.info(f"Retrieving context for query: '{query_text[:150]}...'")
    
    retrieved_docs = GLOBAL_FAISS_MANAGER.query(query_text, k=NUM_RETRIEVED_CHUNKS)
    if not retrieved_docs:
        logger.warning("RAG query returned no documents.")
        state['retrieved_context'] = "No specifically relevant profiles or projects were found in our knowledge base."
        return state

    relevant_profiles = [d for d in retrieved_docs if d.get("doc_type") == "profile"]
    relevant_projects = [d for d in retrieved_docs if d.get("doc_type") == "project"]
    logger.info(f"Retrieved {len(relevant_profiles)} relevant profiles.")
    for p in relevant_profiles[:3]: logger.debug(f"  - Profile Chunk: {p.get('name')}")
    logger.info(f"Retrieved {len(relevant_projects)} relevant projects.")
    for p in relevant_projects[:3]: logger.debug(f"  - Project Chunk: {p.get('name')}")

    context_str = ""
    if relevant_profiles:
        context_str += "--- RELEVANT TEAM PROFILES ---\n"
        for profile in relevant_profiles[:3]:
            context_str += (
                f"\n- Name: {profile.get('name', 'N/A')}\n"
                f"  Role: {profile.get('role', 'N/A')}\n"
                f"  Expertise: {', '.join(profile.get('core_expertise', []))}\n"
            )
    
    if relevant_projects:
        context_str += "\n\n--- RELEVANT PAST PROJECTS ---\n"
        for project in relevant_projects[:3]:
            context_str += (
                f"\n- Project: {project.get('name', 'N/A')}\n"
                f"  Domain: {', '.join(project.get('domain', []))}\n"
                f"  Description: {project.get('description', 'N/A')}\n"
                f"  Tech Stack Used: {', '.join(project.get('tech_stack', []))}\n"
                f"  Key AI Capabilities: {', '.join(project.get('ai_capabilities', []))}\n"
            )

    logger.info("Successfully built structured context for the agent.")
    state['retrieved_context'] = context_str.strip()
    return state

def generate_proposal_from_template(state: ProposalState) -> ProposalState:
    """
    <<< PROMPT ENHANCED: The prompt is now much more powerful because it receives
    structured, reliable context and has clearer instructions.
    """
    relevance_info = ""
    if state.get('relevance_score') is not None:
        relevance_info += f"Relevance Score: {state['relevance_score']:.2f}\n"
    if state.get('closest_profile_name'):
        profile_name = state['closest_profile_name']
        relevance_info += f"Closest Matching Profile: {profile_name}\n"
        guideline = "Write from the perspective of our agency, highlighting the team's combined strength." \
                    if profile_name == 'General Company Profile' else \
                    f"Tailor the proposal to feature the expertise of {profile_name} as the key expert."
        relevance_info += f"Guideline: {guideline}\n"

    prompt = f"""
You are an expert proposal writer for a tech consultancy. Your task is to generate a compelling, human-like proposal for a job by seamlessly integrating our company's relevant experience.

**1. JOB DETAILS:**
- **Job Title:** {state['job_title']}
- **Job Description:** {state['job_description']}

**2. INTERNAL CONTEXT & GUIDELINES:**
{relevance_info}
---
**Our Relevant Experience (Retrieved from Knowledge Base):**
{state['retrieved_context']}
---

**3. YOUR TASK:**
Write the final proposal text. Use the provided template structure for flow, but **DO NOT include placeholders like `{{introduction}}` or section titles which are mentioned in proposal template in the output.**

**CRITICAL INSTRUCTIONS:**
- **Integrate Naturally:** Weave details from the 'RELEVANT PAST PROJECTS' into the proposal as concrete examples. Don't just list projects. Explain *how* a project (and its specific tech stack or AI capabilities) directly proves our ability to handle this client's needs.
- **Human Tone:** Be confident, professional, and helpful. Sound like a human expert connecting past successes to a future project. Avoid generic AI-speak. Keep the use of language and vocuabulary simple and technical.
- **Follow the Guideline:** Adhere strictly to the perspective guideline (agency vs. individual expert).
- **Final Output:** The output MUST be only the clean, final proposal text, ready to send. THE PROPOSAL SHOULD BE CLEAR AND CONCISE AND SHOULD FOLLOW THE TEMPLATE STRUCTURE

**Proposal Template Structure (for flow):**
{state['proposal_template']}

Generate the final, human-like proposal text now:
"""
    
    logger.info("Generating final proposal with enhanced structured context and prompt.")
    logger.debug(f"Final prompt sent to LLM:\n{prompt}")
    final_proposal_text = execute_openai_call(prompt)
    state['final_proposal'] = final_proposal_text
    return state

builder = StateGraph(ProposalState)
builder.add_node("get_job_details", get_job_details)
builder.add_node("load_template", load_template_node) 
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_proposal", generate_proposal_from_template) 

builder.set_entry_point("get_job_details")
builder.add_edge("get_job_details", "load_template")
builder.add_edge("load_template", "retrieve_context")
builder.add_edge("retrieve_context", "generate_proposal")
builder.add_edge("generate_proposal", END)

graph = builder.compile()

@router.post("/agentic-generate-proposal/{job_id}")
def agentic_generate_proposal(job_id: str, overwrite: bool = False, db: Session = Depends(get_db)):
    logger.info(f"Received request to generate proposal for job_id: {job_id} with overwrite: {overwrite}")
    existing_proposal = db.query(Proposal).filter(Proposal.job_id == job_id).first() 
    if existing_proposal and not overwrite:
        logger.info(f"Existing proposal found for job_id: {job_id}. Returning existing.")
        return {"job_id": job_id, "proposal": existing_proposal.proposal_text, "exists": True}

    logger.info(f"Generating new proposal for job_id: {job_id} (overwrite: {overwrite})")
    initial_state = {"job_id": job_id, "db": db}
    final_state = graph.invoke(initial_state)
    proposal_text = final_state.get("final_proposal", "Error: Proposal generation failed.")

    if existing_proposal:
        logger.info(f"Overwriting existing proposal for job_id: {job_id}")
        existing_proposal.proposal_text = proposal_text
    else:
        logger.info(f"Creating new proposal for job_id: {job_id}")
        new_proposal = Proposal(proposal_text=proposal_text, job_id=job_id)
        db.add(new_proposal)
    
    db.commit()
    if existing_proposal:
        db.refresh(existing_proposal)
        return {"job_id": job_id, "proposal": existing_proposal.proposal_text, "exists": True, "overwritten": True}
    else:
        return {"job_id": job_id, "proposal": proposal_text}

@router.put("/save-proposal/{job_id}")
def save_proposal(job_id: str, proposal_data: Dict[str, str], db: Session = Depends(get_db)):
    logger.info(f"Received request to manually save/update proposal for job_id: {job_id}")
    proposal = db.query(Proposal).filter(Proposal.job_id == job_id).first() 
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found for this job.")

    proposal.proposal_text = proposal_data.get("proposal", "")
    db.commit()
    db.refresh(proposal)
    return {"job_id": job_id, "proposal": proposal.proposal_text, "message": "Proposal updated successfully."}
