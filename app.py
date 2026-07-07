import streamlit as st
from PIL import Image
import os
import pandas as pd
from config import Config
from ai_pipeline import process_handwritten_goal, analyze_all_goals
import time

# --- Page Config ---
st.set_page_config(page_title="Hackathon Goal Analyzer", page_icon="🎯", layout="wide")

# --- Custom CSS for beautiful UI ---
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
        border-left: 5px solid #ff4b4b;
    }
    .global-card {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #1890ff;
    }
    .st-emotion-cache-1v0mbdj > img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    h4 {
        margin-bottom: 5px;
        color: #555;
    }
    h2 {
        margin-top: 0px;
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("Configure your AI Agent")

api_key = st.sidebar.text_input("Gemini API Key", value=Config.API_KEY, type="password")

# Provide model choices that match the API key access
model_presets = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest", "gemini-2.0-flash"]
model_choice = st.sidebar.selectbox("Select Model", model_presets)
custom_model = st.sidebar.text_input("Or enter custom model name manually", value="")

final_model = custom_model if custom_model.strip() != "" else model_choice

st.sidebar.markdown("---")
st.sidebar.info("Upload multiple handwritten goals from the hackathon to automatically extract and analyze them globally!")

# --- Main App ---
st.title("🎯 AI Goal Analyzer Dashboard")
st.markdown("**Extract text from messy handwriting and analyze multiple goals instantly using Gemini Multimodal AI.**")

# Accept multiple files!
uploaded_files = st.file_uploader("Upload images of handwritten goals (JPG, PNG)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Analyze All Goals", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please provide a Gemini API Key in the settings.")
        else:
            all_extracted_texts = []
            evaluation_data = [] # For the summary table
            
            with st.spinner("Processing all images... This may take a moment."):
                start_time = time.time()
                
                # --- Individual Analysis ---
                st.markdown("## 📄 Individual Paper Results")
                for i, uploaded_file in enumerate(uploaded_files):
                    image = Image.open(uploaded_file)
                    try:
                        result = process_handwritten_goal(image, final_model, api_key)
                        all_extracted_texts.append(result.corrected_text)
                        
                        # Add to table data
                        evaluation_data.append({
                            "Paper Name": uploaded_file.name,
                            "Category": result.category,
                            "Sentiment": result.sentiment,
                            "Score (1-10)": result.actionability_score,
                            "Extracted Text": result.corrected_text
                        })
                        
                        # Use an expander for each paper to keep UI clean
                        with st.expander(f"Paper {i+1}: {uploaded_file.name} - (Score: {result.actionability_score}/10)", expanded=False):
                            col_img, col_data = st.columns([1, 1.5])
                            with col_img:
                                st.image(image, use_column_width=True)
                            with col_data:
                                st.markdown(f"### 📝 Extracted Text (Raw)")
                                st.info(f"*{result.raw_text}*")
                                
                                st.markdown(f"### 🇬🇧 Corrected & Translated")
                                st.success(result.corrected_text)
                                
                                # Metrics in columns
                                m1, m2, m3 = st.columns(3)
                                with m1:
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left-color: #00cc96;">
                                        <h4>Sentiment</h4>
                                        <h2>{'😊' if result.sentiment.lower() == 'positive' else '😔' if result.sentiment.lower() == 'negative' else '😐'} {result.sentiment}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with m2:
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left-color: #636efa;">
                                        <h4>Category</h4>
                                        <h2>🏷️ {result.category}</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with m3:
                                    score_color = "#00cc96" if result.actionability_score >= 7 else ("#ffa15a" if result.actionability_score >= 4 else "#ef553b")
                                    st.markdown(f"""
                                    <div class="metric-card" style="border-left-color: {score_color};">
                                        <h4>Actionability</h4>
                                        <h2 style='color:{score_color}'>⚡ {result.actionability_score}/10</h2>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                st.markdown("### 💡 Agent Feedback")
                                st.warning(result.feedback)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # --- Global Analysis & Evaluation Dashboard ---
                if len(all_extracted_texts) > 0:
                    st.markdown("---")
                    st.markdown("## 📊 Evaluation Dashboard (All Papers)")
                    
                    # 1. Evaluation Table
                    st.markdown("### 📋 Quick Evaluation Table")
                    st.markdown("Sort and review all papers at a glance.")
                    df = pd.DataFrame(evaluation_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    st.markdown("---")
                    st.markdown("## 🌍 Global Team Synthesis")
                    with st.spinner("Comparing all goals and extracting frequent concepts..."):
                        try:
                            global_result = analyze_all_goals(all_extracted_texts, final_model, api_key)
                            
                            col_concepts, col_objectives = st.columns(2)
                            
                            # 2. Repeated Concepts / Keywords Chart
                            with col_concepts:
                                st.markdown("### 📈 Most Repeated Concepts (Keywords)")
                                if global_result.top_keywords:
                                    # Create a dataframe for the bar chart
                                    kw_data = {"Keyword": [k.keyword for k in global_result.top_keywords], 
                                               "Frequency": [k.frequency for k in global_result.top_keywords]}
                                    kw_df = pd.DataFrame(kw_data).set_index("Keyword")
                                    st.bar_chart(kw_df)
                                else:
                                    st.info("No common keywords found.")
                                    
                            # 3. Frequent Objectives List
                            with col_objectives:
                                st.markdown("""
                                <div class="global-card">
                                    <h3>🎯 Top Common Objectives</h3>
                                    <ul>
                                """ + "".join([f"<li>{obj}</li>" for obj in global_result.most_frequent_objectives]) + """
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # 4. Feedback & Comparison
                            col_comp, col_feed = st.columns(2)
                            with col_comp:
                                st.info("### ⚖️ Comparison Summary\n" + global_result.comparison_summary)
                            with col_feed:
                                st.success("### 📣 Overall Team Feedback\n" + global_result.overall_feedback)
                                
                        except Exception as e:
                            st.error(f"Error performing global analysis: {str(e)}")
                            
                end_time = time.time()
                st.success(f"Batch Analysis completed in {end_time - start_time:.2f} seconds!")
