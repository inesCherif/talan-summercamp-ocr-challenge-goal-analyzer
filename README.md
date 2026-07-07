# 🎯 Challenge Goal Analyzer Dashboard

Welcome to the **Hackathon Goal Analyzer**, an AI-powered dashboard built for the Talan Summercamp hackathon. This application automates the process of reading handwritten goals from participants, analyzing them, and generating a global synthesis for the entire team.

## ✨ Features
* **Advanced OCR & Multimodal AI:** Uses Google Gemini 2.5 Flash to perfectly extract text from images, even with messy handwriting or bad grammar.
* **Auto-Translation & Correction:** Automatically translates French to English and corrects grammar mistakes in the extracted text.
* **Granular NLP Analysis:** Evaluates each paper for:
  * **Sentiment:** (Positive / Negative / Neutral)
  * **Domain/Category:** (e.g., Technical, Soft Skills, Team)
  * **Actionability Score:** A realistic rating from 1 to 10.
  * **Personalized Agent Feedback:** Constructive advice for that specific goal.
* **Batch Processing:** Drag and drop multiple images at once to process the entire team's goals in one go.
* **Interactive Evaluation Dashboard:** An interactive data table summarizing all participants' scores, sentiments, and texts for quick sorting and evaluation.
* **Global Team Synthesis:** Automatically analyzes the entire batch of papers to generate:
  * A bar chart of the most repeated concepts and keywords.
  * A list of the most frequent team objectives.
  * A comparison summary and overall feedback for the group.

---

## 🏗️ Architecture & Files

The project is built using a highly modular architecture for speed and accuracy:

* **`app.py`** (The UI Layer)
  * Built with **Streamlit** for a fast, interactive, and beautiful dashboard.
  * Handles the multi-file drag-and-drop upload, rendering the evaluation dataframe, generating the bar charts for keywords, and displaying the individual and global metrics in clean, collapsible sections.

* **`ai_pipeline.py`** (The Intelligence & Orchestration Layer)
  * Powered by **LangChain** and **Google Gemini (GenAI API)**.
  * Uses **Pydantic** to enforce Structured Outputs, ensuring the LLM always replies with strict, predictable JSON data (Sentiment, Score, Keywords) rather than unstructured chat.
  * Contains `process_handwritten_goal` for individual OCR/analysis and `analyze_all_goals` for the global team synthesis.

* **`config.py`** (The Configuration Layer)
  * Stores the environment variables and default AI model settings. 
  * Safely manages the Gemini API Key and allows you to easily switch between models like `gemini-2.5-flash` or `gemini-2.5-pro`.

* **`requirements.txt`** (Dependencies)
  * Contains the required Python packages (`streamlit`, `langchain`, `langchain-google-genai`, `python-dotenv`, `pillow`, `pydantic`, `pandas`).

---

## 🚀 How to Run the Application

### 1. Prerequisites
Make sure you have Python installed. It is recommended to use a virtual environment.

### 2. Install Dependencies
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### 3. API Key Setup
The app requires a Google Gemini API Key. You can either:
- Create a `.env` file in the project root and add: `GEMINI_API_KEY=your_api_key_here`
- OR enter the API key directly into the sidebar of the application once it's running.

### 4. Start the Dashboard
Run the following command to start the Streamlit server:
```bash
python -m streamlit run app.py
```
This will automatically open the dashboard in your default web browser (usually at `http://localhost:8501`).

---

## 💡 Usage Guide
1. Start the app and ensure your API key and chosen Model (e.g., `gemini-2.5-flash`) are set in the sidebar.
2. Take photos of the handwritten hackathon goals (JPG/PNG).
3. Drag and drop all the images into the main upload area.
4. Click **"🚀 Analyze All Goals"**.
5. Wait a few seconds for the agent to process the batch.
6. Review the individual extracted texts, sort the Evaluation Table, and read the Global Team Synthesis!
