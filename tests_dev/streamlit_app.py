import streamlit as st 
from workflow_agents_HR import WorkforceState, app
import base64
import io
from PIL import Image
from loguru import logger
import os



st.set_page_config(layout="wide") 

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
    trend1, trend2, trend3, trend4 = st.columns(4)
    with trend1:
        st.write("**Skill Name**")
    with trend2:
        st.write("**Demand**")
    with trend3:
        st.write("**Growth Rate**")
    with trend4:
        st.write("**Category**")
    for trend in trends:
        with trend1:
            st.write(trend["skill_name"])
        with trend2:
            st.write(trend["demand_level"])
        with trend3:
            st.write(trend["growth_rate"])
        with trend4:
            st.write(trend["category"])

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
    st.subheader("Report:")
    logger.debug(f"Type report : {type(report)}")
    st.markdown(report)