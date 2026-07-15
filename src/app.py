import os
import streamlit as st
from PIL import Image
from retriever import FashionRetriever

# Page configuration
st.set_page_config(
    page_title="Fashion Retrieval Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, professional dark-themed research dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header styles */
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #21262d;
    }
    
    .main-title {
        font-weight: 800;
        font-size: 2.5rem;
        color: #f0f6fc;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        font-weight: 300;
    }
    
    /* Card layout */
    .result-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #30363d;
        margin-bottom: 24px;
        display: flex;
        flex-direction: column;
    }
    
    .result-img-container {
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 12px;
        border: 1px solid #30363d;
    }
    
    /* Table styling for attributes */
    .attr-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 10px;
        color: #c9d1d9;
    }
    
    .attr-table td {
        padding: 6px 4px;
        border-bottom: 1px solid #21262d;
    }
    
    .attr-table td.label-cell {
        font-weight: 600;
        color: #8b949e;
        width: 40%;
    }
    
    .attr-table td.val-cell {
        text-transform: capitalize;
        color: #f0f6fc;
        text-align: right;
    }
    
    /* Score display */
    .score-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #30363d;
    }
    
    .hybrid-score {
        font-size: 1.15rem;
        font-weight: 800;
        color: #58a6ff;
    }
    
    .breakdown-text {
        font-size: 0.78rem;
        color: #8b949e;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Retriever (cached)
@st.cache_resource
def get_retriever():
    return FashionRetriever()

try:
    retriever = get_retriever()
    initialized = True
except Exception as e:
    st.error(f"Error loading model database. Please run the indexing pipeline. Detail: {e}")
    initialized = False

# Layout header
st.markdown("""
<div class="title-container">
    <div class="main-title">Multimodal Fashion & Context Retrieval</div>
    <div class="subtitle">Search engine using CLIP embeddings and spatial-crop zero-shot attribute classification</div>
</div>
""", unsafe_allow_html=True)

if initialized:
    # Sidebar
    st.sidebar.markdown("<h3 style='color: #f0f6fc; font-weight:800; margin-top:0;'>Search Configuration</h3>", unsafe_allow_html=True)
    top_k = st.sidebar.slider("Number of top matches (k)", min_value=1, max_value=20, value=6)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #f0f6fc; font-weight:600;'>Catalog Filters</h4>", unsafe_allow_html=True)
    filter_upper_color = st.sidebar.selectbox("Upper Color", ["All"] + retriever.colors)
    filter_upper_type = st.sidebar.selectbox("Upper Type", ["All"] + retriever.upper_types)
    filter_lower_color = st.sidebar.selectbox("Lower Color", ["All"] + retriever.colors)
    filter_lower_type = st.sidebar.selectbox("Lower Type", ["All"] + retriever.lower_types)
    filter_env = st.sidebar.selectbox("Environment Setting", ["All"] + retriever.environments)

    # Search Bar
    query = st.text_input("Enter natural language query description:", value="A person in a bright yellow raincoat.")
    
    if query:
        # Search DB
        with st.spinner("Executing retrieval queries..."):
            results, targets = retriever.search(query, top_k=50)
            
        # Display Parsed query targets
        st.markdown("<div style='background-color:#161b22; padding:12px; border-radius:6px; border:1px solid #30363d; margin-bottom: 20px; font-size:0.9rem;'>", unsafe_allow_html=True)
        st.markdown(f"**Query Parsing Breakdown:** upper: `{targets['upper_color'] or 'any'} {targets['upper_type'] or 'any'}` | lower: `{targets['lower_color'] or 'any'} {targets['lower_type'] or 'any'}` | tie: `{'yes (' + (targets['tie_color'] or 'any') + ')' if targets['has_tie'] else 'any'}` | setting: `{targets['environment'] or 'any'}`", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply Sidebar Filters
        filtered_results = []
        for res in results:
            payload = res["payload"]
            
            if filter_upper_color != "All" and payload["upper_color"] != filter_upper_color:
                continue
            if filter_upper_type != "All" and payload["upper_type"] != filter_upper_type:
                continue
            if filter_lower_color != "All" and payload["lower_color"] != filter_lower_color:
                continue
            if filter_lower_type != "All" and payload["lower_type"] != filter_lower_type:
                continue
            if filter_env != "All" and payload["environment"] != filter_env:
                continue
                
            filtered_results.append(res)
            
        display_results = filtered_results[:top_k]
        
        if not display_results:
            st.warning("No images match the selected sidebar filters.")
        else:
            # Display results in columns
            cols_per_row = 3
            rows = (len(display_results) + cols_per_row - 1) // cols_per_row
            
            for row_idx in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    item_idx = row_idx * cols_per_row + col_idx
                    if item_idx < len(display_results):
                        res = display_results[item_idx]
                        payload = res["payload"]
                        
                        with cols[col_idx]:
                            # Outer card
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            
                            # Render image
                            img_path = res["path"]
                            if os.path.exists(img_path):
                                img = Image.open(img_path)
                                st.image(img, use_container_width=True)
                            else:
                                st.markdown(f"<div style='height:200px; background-color:#21262d; display:flex; align-items:center; justify-content:center; color:#8b949e;'>Image {res['filename']} not found</div>", unsafe_allow_html=True)
                                
                            # Attributes Table
                            tie_desc = payload['tie_color'] if payload['has_tie'] else 'No'
                            table_html = f"""
                            <table class="attr-table">
                                <tr>
                                    <td class="label-cell">Upper body</td>
                                    <td class="val-cell">{payload['upper_color']} {payload['upper_type']}</td>
                                </tr>
                                <tr>
                                    <td class="label-cell">Lower body</td>
                                    <td class="val-cell">{payload['lower_color']} {payload['lower_type']}</td>
                                </tr>
                                <tr>
                                    <td class="label-cell">Necktie</td>
                                    <td class="val-cell">{tie_desc}</td>
                                </tr>
                                <tr>
                                    <td class="label-cell">Environment</td>
                                    <td class="val-cell">{payload['environment']}</td>
                                </tr>
                                <tr>
                                    <td class="label-cell">Style Vibe</td>
                                    <td class="val-cell">{payload['style']}</td>
                                </tr>
                            </table>
                            """
                            st.markdown(table_html, unsafe_allow_html=True)
                            
                            # Score and breakdown
                            st.markdown(f"""
                            <div class="score-row">
                                <div class="hybrid-score">Score: {res['hybrid_score']:.3f}</div>
                                <div class="breakdown-text">
                                    CLIP: {res['clip_score']:.3f}<br>
                                    Boost: +{res['boost_score']:.3f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Database not populated. Please run indexer pipeline script.")
