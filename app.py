import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import json
from datetime import datetime

# Import custom modules (make sure they're in the same directory)
from youtube_collector import YouTubeDataCollector
from content_research_agent import ContentResearchAgent
from thumbnail_ctr_model import ThumbnailCTRPredictor

# Configure page
st.set_page_config(
    page_title="YouTube Viral Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF0000, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FF0000;
    }
    .recommendation {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'thumbnail_predictor' not in st.session_state:
    st.session_state.thumbnail_predictor = None
if 'research_agent' not in st.session_state:
    st.session_state.research_agent = None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=80)
    st.title("🎬 Viral Predictor")
    
    st.markdown("---")
    
    # Niche selection
    niche = st.selectbox(
        "Select Your Niche",
        ["Tech", "Finance"],
        key="niche_selector"
    )
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎯 Thumbnail Analyzer", "🤖 Content Copilot", "📊 Dataset Viewer"],
        key="navigation"
    )
    
    st.markdown("---")
    
    # API Configuration
    with st.expander("⚙️ API Configuration"):
        youtube_api = st.text_input("YouTube API Key", type="password")
        groq_api = st.text_input("Groq API Key", type="password")
       
        
        if st.button("Save Configuration"):
            st.success("✅ Configuration saved!")
    
    st.markdown("---")
    st.caption("Made with ❤️ for YouTube Creators")

# Main content
if page == "🏠 Dashboard":
    st.markdown("<h1 class='main-header'>YouTube Viral Predictor Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("### Your AI-Powered Content Strategy Hub")
    
    # Check if dataset exists
    if os.path.exists('data/youtube_dataset.csv'):
        df = pd.read_csv('data/youtube_dataset.csv')
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Total Videos", len(df))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Views", f"{df['view_count'].mean():,.0f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Top Virality", f"{df['virality_score'].max():.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Avg Engagement", f"{df['engagement_rate'].mean():.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Virality Score Distribution")
            fig = px.histogram(df, x='virality_score', nbins=30, 
                             color_discrete_sequence=['#FF0000'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("👁️ Views vs Engagement")
            fig = px.scatter(df, x='view_count', y='engagement_rate',
                           size='like_count', hover_data=['title'],
                           color='virality_score', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top performing videos
        st.subheader("🔥 Top 5 Viral Videos")
        top_videos = df.nlargest(5, 'virality_score')[['title', 'view_count', 'engagement_rate', 'virality_score']]
        st.dataframe(top_videos, use_container_width=True)
        
    else:
        st.warning("📊 No dataset found. Please collect YouTube data first!")
        
        if st.button("🔍 Collect Data Now"):
            with st.spinner("Collecting YouTube data..."):
                st.info("Feature coming soon! Run `youtube_collector.py` script to collect data.")

elif page == "🎯 Thumbnail Analyzer":
    st.markdown("<h1 class='main-header'>Thumbnail CTR Predictor</h1>", unsafe_allow_html=True)
    st.markdown("### Upload your thumbnail and get instant virality insights")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Thumbnail (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload your video thumbnail for analysis"
        )
        
        if uploaded_file:
            # Save uploaded file
            os.makedirs('temp', exist_ok=True)
            img_path = f'temp/{uploaded_file.name}'
            
            with open(img_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Display image
            st.image(uploaded_file, caption="Your Thumbnail", width=400)  # set desired width

            
            if st.button("🔮 Analyze Thumbnail", type="primary"):
                with st.spinner("Analyzing your thumbnail..."):
                    # Placeholder for actual prediction
                    # predictor = ThumbnailCTRPredictor()
                    # predictor.load_model()
                    # result = predictor.predict_ctr(img_path)
                    
                    # Mock result for demo
                    result = {
                        'ctr_score': 78.5,
                        'features': {
                            'has_face': 1,
                            'mean_brightness': 145,
                            'mean_saturation': 92,
                            'text_density': 0.15,
                            'num_faces': 1,
                            'contrast': 68
                        },
                        'recommendations': [
                            "Great use of faces! Keep it up.",
                            "Consider increasing saturation slightly for more pop",
                            "Text density is perfect - not too cluttered"
                        ]
                    }
                    
                    st.session_state.analysis_result = result
    
    with col2:
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            
            # CTR Score gauge
            st.markdown("### 🎯 Virality Score")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['ctr_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CTR Prediction"},
                delta={'reference': 65},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature breakdown
            st.markdown("### 📊 Feature Analysis")
            features_df = pd.DataFrame([result['features']]).T
            features_df.columns = ['Value']
            st.dataframe(features_df, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            for rec in result['recommendations']:
                st.markdown(f"<div class='recommendation'>✓ {rec}</div>", unsafe_allow_html=True)

elif page == "🤖 Content Copilot":
    st.markdown("<h1 class='main-header'>AI Content Copilot</h1>", unsafe_allow_html=True)
    st.markdown("### Get trending script ideas powered by AI")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### ⚙️ Research Settings")
        
        research_niche = st.selectbox("Niche", ["tech", "finance"], key="research_niche")
        num_outlines = st.slider("Number of Outlines", 3, 10, 5)
        
        if st.button("🚀 Generate Ideas", type="primary"):
            with st.spinner("Researching trends and generating ideas..."):
                # Placeholder for actual research
                st.session_state.generated_outlines = """
## 1. "I Built an AI That Reads Your Mind (It Actually Works!)"

**Hook:** "What if I told you AI can now predict your thoughts with 87% accuracy?"

**Main Points:**
- Demo of mind-reading AI
- The science behind neural networks
- Ethical implications
- How YOU can try it

**CTA:** Download the model from GitHub

**Watch Time:** 12 minutes

**Virality Potential:** 9/10 - Combines curiosity, demo, and controversy

---

## 2. "The Programming Language Everyone Will Use in 2025"

**Hook:** "Python is dying... and this language is taking over"

**Main Points:**
- Current trends in programming
- Deep dive into the new language
- Real-world use cases
- Career opportunities

**CTA:** Start learning with my free course

**Watch Time:** 15 minutes

**Virality Potential:** 8/10 - Taps into FOMO and career anxiety

---

## 3. "I Automated My Entire Job with AI (Boss Doesn't Know)"

**Hook:** "I work 2 hours a week... thanks to these AI tools"

**Main Points:**
- Tools I use daily
- Automation workflow
- Time saved metrics
- Ethical considerations

**CTA:** Join my automation community

**Watch Time:** 10 minutes

**Virality Potential:** 9/10 - Perfect for lazy + ambitious viewers
"""
    
    with col1:
        if 'generated_outlines' in st.session_state:
            st.markdown("### 📝 Generated Script Outlines")
            st.markdown(st.session_state.generated_outlines)
            
            if st.button("💾 Download as Text"):
                st.download_button(
                    "Download",
                    st.session_state.generated_outlines,
                    file_name=f"script_outlines_{datetime.now().strftime('%Y%m%d')}.txt"
                )
        else:
            st.info("👈 Configure settings and click 'Generate Ideas' to start")

elif page == "📊 Dataset Viewer":
    st.markdown("<h1 class='main-header'>YouTube Dataset Explorer</h1>", unsafe_allow_html=True)
    
    if os.path.exists('data/youtube_dataset.csv'):
        df = pd.read_csv('data/youtube_dataset.csv')
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_views = st.number_input("Min Views", 0, int(df['view_count'].max()), 0)
        with col2:
            min_engagement = st.number_input("Min Engagement %", 0.0, float(df['engagement_rate'].max()), 0.0)
        with col3:
            min_virality = st.number_input("Min Virality Score", 0, 100, 0)
        
        # Apply filters
        filtered_df = df[
            (df['view_count'] >= min_views) &
            (df['engagement_rate'] >= min_engagement) &
            (df['virality_score'] >= min_virality)
        ]
        
        st.markdown(f"### 📊 Showing {len(filtered_df)} videos")
        
        # Display data
        st.dataframe(
            filtered_df[['title', 'channel_title', 'view_count', 'engagement_rate', 'virality_score']],
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv,
            "filtered_youtube_data.csv",
            "text/csv"
        )
    else:
        st.warning("No dataset found!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎬 YouTube Viral Predictor | Built with Streamlit, TensorFlow & LangChain</p>
    <p>Need help? Check out the <a href='https://github.com/yourusername/viral-predictor'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)