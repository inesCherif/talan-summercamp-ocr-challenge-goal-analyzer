from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import List
from PIL import Image
import base64
import io

class GoalAnalysis(BaseModel):
    raw_text: str = Field(description="The exact text extracted from the image, preserving the original language (French or English) and any bad grammar.")
    corrected_text: str = Field(description="The grammar-corrected and translated (to English) version of the text.")
    sentiment: str = Field(description="The sentiment of the goal (Positive, Negative, or Neutral).")
    category: str = Field(description="The category/domain of the goal (e.g., Technical, Soft Skills, Personal, Team, Career).")
    actionability_score: int = Field(description="Score from 1 to 10 indicating how actionable and realistic the goal is.")
    feedback: str = Field(description="A short, encouraging sentence or constructive feedback about the goal.")

class KeywordFrequency(BaseModel):
    keyword: str = Field(description="The repeated concept, word, or skill.")
    frequency: int = Field(description="Approximate number of times this concept appeared across the goals.")

class GlobalAnalysis(BaseModel):
    most_frequent_objectives: List[str] = Field(description="List of the 3-5 most common or recurring themes/objectives among all the goals provided.")
    top_keywords: List[KeywordFrequency] = Field(description="The top 5-10 most repeated concepts, words, or skills mentioned.")
    comparison_summary: str = Field(description="A short paragraph comparing the goals. Are they mostly technical? Are they aligned with each other? Where do they differ?")
    overall_feedback: str = Field(description="Global feedback for the entire group or team based on all the submitted goals.")

def get_llm(model_name: str, api_key: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2, 
    )

def process_handwritten_goal(image: Image.Image, model_name: str, api_key: str) -> GoalAnalysis:
    """
    Takes an image, extracts text, and analyzes it in one shot using Gemini.
    """
    llm = get_llm(model_name, api_key)
    structured_llm = llm.with_structured_output(GoalAnalysis)
    
    buffered = io.BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    prompt = """
    You are an expert OCR system and a supportive hackathon mentor.
    Please read the attached image of a handwritten goal (it might be in English or French, and might have bad grammar/handwriting).
    
    1. Extract the raw text exactly as written.
    2. Correct the grammar and translate it to English.
    3. Analyze the sentiment (Positive, Negative, or Neutral).
    4. Categorize the domain of the goal.
    5. Score its actionability from 1 to 10 while  taking into consideration the program of talan includes: AI ML deeplearning transformer LLM prompt engineering agentic AI , genAi
Cybersecurity, blockchain, quantum ai , motivation, team spirit , curiosity.
    6. Provide a short, constructive feedback sentence.
    """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_str}"},
            },
        ]
    )
    
    response = structured_llm.invoke([message])
    return response


def analyze_all_goals(extracted_texts: List[str], model_name: str, api_key: str) -> GlobalAnalysis:
    """
    Takes a list of all extracted texts and performs a global analysis to find frequent objectives and compare them.
    """
    llm = get_llm(model_name, api_key)
    structured_llm = llm.with_structured_output(GlobalAnalysis)
    
    combined_texts = "\n\n".join([f"Goal {i+1}:\n{text}" for i, text in enumerate(extracted_texts)])
    
    prompt = f"""
    You are an expert mentor reviewing the goals of multiple participants in a hackathon (Talan Summercamp).
    Here are the goals submitted by the participants:
    
    {combined_texts}
    
    Please analyze all of these goals collectively:
    1. Identify the most frequent and recurring objectives or themes.
    2. Extract the top 5-10 most repeated concepts, words, or skills and estimate their frequency.
    3. Write a short comparison summary between the goals.
    4. Provide overall constructive feedback for the entire group.
    """
    
    message = HumanMessage(content=[{"type": "text", "text": prompt}])
    
    response = structured_llm.invoke([message])
    return response
