from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai
import pypdf
import docx
import io
import json
import shutil
import tempfile
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# Get the Uvicorn logger
logger = logging.getLogger("uvicorn")

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
except KeyError:
    raise RuntimeError("GEMINI_API_KEY environment variable not set.")

app = FastAPI()

# --- Pydantic Models for Request Bodies ---
class QuestionRequest(BaseModel):
    difficulty: str

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    difficulty: str

class SummaryRequest(BaseModel):
    chatHistory: List[dict]

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request received: {request.method} {request.url.path}")
    response = await call_next(request)
    return response

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    if not (file.content_type == "application/pdf" or file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF or DOCX file.")

    tmp_file_path = None
    try:
        suffix = ".pdf" if file.content_type == "application/pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
        
        resume_text = ""
        if file.content_type == "application/pdf":
            pdf_reader = pypdf.PdfReader(tmp_file_path)
            for page in pdf_reader.pages:
                resume_text += page.extract_text() or ""
        else:
            doc = docx.Document(tmp_file_path)
            for para in doc.paragraphs:
                resume_text += para.text + "\n"

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""Extract the full name, email address, and phone number from the following resume text. Return it as a valid JSON object with keys 'name', 'email', 'phone'. If a field is not found, its value should be null. Resume Text: --- {resume_text} ---"""
        
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        
        return json.loads(cleaned_response)

    except Exception as e:
        logger.error(f"Error processing resume: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {str(e)}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

@app.post("/generate-question")
async def generate_question(request: QuestionRequest):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"You are a senior technical interviewer. Generate a single, specific, {request.difficulty} level interview question for a full-stack developer role focusing on React and Node.js. Return a JSON object with a single key 'question'."
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_response)
    except Exception as e:
        logger.error(f"Error generating question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate question: {str(e)}")

@app.post("/evaluate-answer")
async def evaluate_answer(request: EvaluationRequest):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""You are an expert technical interviewer. Evaluate the following answer to the given question. Provide a score from 1-10 and a brief one-sentence justification for the score. The question difficulty was {request.difficulty}. Question: '{request.question}' Answer: '{request.answer}'. Return a JSON object with keys 'score' and 'justification'."""
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_response)
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {str(e)}")

@app.post("/generate-summary")
async def generate_summary(request: SummaryRequest):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Convert chat history to a more readable format for the prompt
        transcript = "\n".join([f"Q: {item['q']}\nA: {item['a']}" for item in request.chatHistory])
        prompt = f"""Based on this entire interview transcript, provide a final score out of 100 and a 2-3 sentence summary of the candidate's strengths and weaknesses. Transcript: {transcript}. Return a JSON object with keys 'finalScore' and 'finalSummary'."""
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_response)
    except Exception as e:
        logger.error(f"Error generating summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
