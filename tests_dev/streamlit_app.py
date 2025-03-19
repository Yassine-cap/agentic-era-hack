import streamlit as st 
from workflow_agents_HR import WorkforceState, app
import base64
import io
from PIL import Image
from loguru import logger
import os
import json
import pandas as pd
import re

st.set_page_config(layout="wide") 


def extract_json_from_response(response):
    # Utiliser une expression régulière pour trouver le JSON dans la réponse
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        # Supprimer les commentaires ou notes entre parenthèses
        json_str = re.sub(r'\s*\([^)]*\)', '', json_str)
        try:
            # Charger la chaîne JSON en tant que dictionnaire Python
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON: {e}")
            return None
    else:
        print("JSON non trouvé dans la réponse")
        return None


def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def get_title(title):
    logo_path = os.path.join("tests_dev/image", "Isotype_Onepoint_rvb_foncé.png")
    if os.path.exists(logo_path):
        logo_base64 = get_base64_image(logo_path)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 50px; margin-right: 10px;">
                <h1>{title}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("L'image du logo ne peut pas être trouvée. Veuillez vérifier le chemin.")
        
def get_side_bar_logo():
    logo_path = os.path.join("tests_dev/image", "Logo_Onepoint+baseline_rvb_foncé.png")
    image = Image.open(logo_path)
    st.sidebar.image(image, caption=None, output_format="JPEG")
    second_logo_path = os.path.join("tests_dev/image", "google_logo.png")
    second_image = Image.open(second_logo_path)
    st.sidebar.image(second_image, caption=None, output_format="JPEG")


get_title("TalentForge – Shaping the workforce of tomorrow")
st.info("**Our agent offers actionable insights to help businesses transition smoothly.**")
get_side_bar_logo()
st.sidebar.markdown(
    """
    **Core Functionalities and Features:**
    - Real-Time Makert Intelligence ⏳📈
    - Actionable Insights & Metrics 🎯📊 
    - Personalized Reskilling Recommendations 🎓✨
    - Interactive Dashboard Visualization 📊🖥️
    - Scalable & Adaptable Solution 📡⚙️
    """
)
st.markdown("---")


query = st.chat_input("Tell me workforce domain you're interested in")

if query:
    initial_state: WorkforceState = {
        "query": query,
        "search_results": [],
        "trends": [],
        "aggregated_data": {},
        "visualizations": [],
        "report": "",
        "status": "initialized",
        "error": None
    }
    with st.spinner("Calling the agent..."):
        final_state = app.invoke(initial_state)
    st.subheader("Trends:")
    trends = final_state["trends"]
    trends_df = pd.DataFrame(trends)
    st.dataframe(trends_df)

    st.subheader("Visualizations:")
    
    col1, col2 = st.columns(2)
    vizualisations = final_state["visualizations"]
    for i, viz in enumerate(vizualisations):
        if viz["type"] == "image":
            img_data = viz["data"]
            if isinstance(img_data, str):
                try:
                    # Decode Base64 string
                    img_bytes = base64.b64decode(img_data)
                    img = Image.open(io.BytesIO(img_bytes))
                    # st.image(img, caption=viz.get("title", ""))
                    logger.info("img_data is str")
                except Exception as e:
                    st.error(f"Error decoding image: {e}")

            # Handle already opened image
            elif isinstance(img_data, Image.Image):
                logger.info("img_data is Image")
                img = img_data
                # st.image(img_data, caption=viz.get("title", ""))
            
            else:
                st.error("Unsupported image format")
                continue
            if i % 2 == 0:
                col1.image(img, caption=viz.get("title", ""))
            else:
                col2.image(img, caption=viz.get("title", ""))

            
    
    report = final_state["report"]
    try:
        report_json = extract_json_from_response(report)
        executive_summary = report_json.get("executive_summary", "")
        st.subheader("Executive Summary:")
        st.markdown(executive_summary)
        
        st.subheader("Industry Insights:")
        industry_insights = report_json.get("industry_insights", {})
        key_trends = industry_insights.get("key_trends", [])
        
        st.write("**Key Trends:**")
        key_trends_df = pd.DataFrame(key_trends)
        key_trends_df.rename(columns={'title': 'Key Trend'}, inplace=True)
        st.dataframe(key_trends_df)
        
        st.write("**Challenges:**")
        challenges = industry_insights.get("challenges", [])
        challenges_df = pd.DataFrame(challenges)
        challenges_df.rename(columns={'title': 'Challenges'},
                              inplace=True)
        st.dataframe(challenges_df)
        
        skills_demand = report_json.get("skills_demand", {})
        
        technical_skills = skills_demand.get("technical_skills", [])
        technical_skills_df = pd.DataFrame(technical_skills)
        technical_skills_df.rename(columns={'name': 'Technical Skill'}, inplace=True)
        
        cognitive_skills = skills_demand.get("cognitive_skills", [])
        cognitive_skills_df = pd.DataFrame(cognitive_skills)
        cognitive_skills_df.rename(columns={'name': 'Cognitive Skill'}, inplace=True)
        
        soft_skills = skills_demand.get("soft_skills", [])
        soft_skills_df = pd.DataFrame(soft_skills)
        soft_skills_df.rename(columns={'name': 'Soft Skill'}, inplace=True)
        
        workforce_gaps = report_json.get("workforce_gaps", {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Technical Skills")
            st.dataframe(technical_skills_df)
            technical_gap = workforce_gaps.get("technical", {})
            st.info(technical_gap)
        with col2:
            st.subheader("Cognitive Skills")
            st.dataframe(cognitive_skills_df)
            cognitive_gap = workforce_gaps.get("cognitive", {})
            st.info(cognitive_gap)
        with col3:
            st.subheader("Soft Skills")
            st.dataframe(soft_skills_df)
            soft_gap = workforce_gaps.get("soft_skills", {})
            st.info(soft_gap)
        
        reskilling_recommendations = report_json.get("reskilling_recommendations", [])
        st.subheader("Reskilling Recommendations:")
        technical_reskilling = reskilling_recommendations.get("technical_skills", [])
        technical_reskilling_df = pd.DataFrame(technical_reskilling)
        technical_reskilling_df.rename(columns={'name': 'Technical Skill'}, inplace=True)
        st.write("**Technical Reskilling and training recommendations**")
        st.dataframe(technical_reskilling_df)
        
        cognitive_reskilling = reskilling_recommendations.get("cognitive_skills", [])
        cognitive_reskilling_df = pd.DataFrame(cognitive_reskilling)
        cognitive_reskilling_df.rename(columns={'name': 'Cognitive Skill'}, inplace=True)
        st.write("**Cognitive Reskilling and training recommendations**")
        st.dataframe(cognitive_reskilling_df)
        
        soft_reskilling = reskilling_recommendations.get("soft_skills", [])
        soft_reskilling_df = pd.DataFrame(soft_reskilling)
        soft_reskilling_df.rename(columns={'name': 'Soft Skill'}, inplace=True)
        st.write("**Soft Reskilling and training recommendations**")
        st.dataframe(soft_reskilling_df)
        
        conclusion = report_json.get("conclusion", "")
        st.subheader("Conclusion:")
        st.write(conclusion)
        
        st.subheader("Sources:")
        
        sources = final_state["search_results"]
        for source in sources:
            st.markdown(f"- [{source['title']}]({source['url']})")
        
    except Exception as e:
        logger.error(f"Error parsing report: {e}")
        report = "Unable to parse the report. Please check the logs for more details."
    