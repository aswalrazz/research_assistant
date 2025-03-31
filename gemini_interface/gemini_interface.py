import os, sys
from os.path import dirname as up

sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import google.generativeai as genai
from dotenv import load_dotenv

from research_paper_analyst import RESEARCH_PAPER, FOLLOWUP_CONTEXT
from typing import Generator

_ = load_dotenv()


def initialize_gemini(model_name: str):
    """Initialize the Gemini model with the given model_name."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        raise EOFError(f"Failed to initialize Gemini model: {str(e)}")


def analyze_pdf_content(model, pdf_content: str) -> str:
    """Analyze the given PDF content with the Gemini model."""
    try:
        response = model.generate_content(
            [{"mime_type": "application/pdf", "data": pdf_content}, RESEARCH_PAPER],
            stream=True,
        )
        for chunk in response:
            yield chunk
    except Exception as e:
        raise RuntimeError(f"Error analyzing PDF content: {str(e)}")


def process_query_stream(model, pdf_content: str, notes: str, query: str):
    """Process a user query against the analyzed PDF content using the Gemini model."""
    try:
        context = f"{notes}\n\n{FOLLOWUP_CONTEXT}\n\n{query}"
        response = model.generate_content(
            [{"mime_type": "application/pdf", "data": pdf_content}, context],
            stream=True,
        )
        for chunk in response:
            yield chunk
    except Exception as e:
        yield f"Error processing query: {str(e)}"
