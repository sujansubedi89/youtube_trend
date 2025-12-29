# 🎬 YouTube Viral Predictor & Content Copilot

An AI-powered tool that helps YouTube creators predict video virality and generate content ideas using machine learning and trend analysis.

> **⚠️ Note:** This is a **proof-of-concept/educational project**. The predictions are based on historical data patterns and should be used as inspiration rather than absolute metrics. Real virality depends on many dynamic factors not captured in this simplified model.
<img width="1833" height="871" alt="image" src="https://github.com/user-attachments/assets/ace70a22-93ee-42f7-acda-9f042155b7fe" />

<img width="1918" height="1028" alt="image" src="https://github.com/user-attachments/assets/eb174178-c2b2-4f6e-aa61-4c73c0fb90c5" />

## 🌟 Features

- **📊 Thumbnail CTR Predictor**: Upload a thumbnail and get an estimated click-through rate score based on visual features (faces, colors, text density)
- **🤖 AI Content Copilot**: Generate 5 viral video script ideas using Google Trends and AI analysis
- **📈 YouTube Analytics Dashboard**: View trending videos in your niche with engagement metrics
- **🎯 Niche Focus**: Optimized for Tech and Finance content creators

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- YouTube Data API key (free from Google Cloud Console)
- Groq API key (free from console.groq.com)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/youtube-viral-predictor.git
cd youtube-viral-predictor
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys**

Create a `.env` file in the project root:
```bash
YOUTUBE_API_KEY=your_youtube_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📋 Getting API Keys

### YouTube Data API (Free)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials → API Key
5. Copy your key to `.env`

**Free Tier**: 10,000 units/day (~100 video details/day)

### Groq API (Free)
1. Go to [Groq Console](https://console.groq.com/)
2. Sign up with Google/GitHub
3. Create an API key
4. Copy your key to `.env`

**Free Tier**: 30 requests/minute with Llama 70B model

---

## 📊 Usage

### 1. Collect YouTube Data
```bash
python youtube_collector.py
```
This scrapes YouTube videos in your niche and downloads thumbnails to `data/` folder.

### 2. Generate Content Ideas
```bash
python content_research_agent.py
```
This analyzes Google Trends and generates AI-powered script outlines.

### 3. Train Thumbnail Model (Optional)
```bash
python thumbnail_ctr_model.py
```
Trains a CNN model on collected thumbnails. **Note**: Requires substantial data (100+ videos) for meaningful results.

### 4. Launch Dashboard
```bash
streamlit run app.py
```
Access the web interface to:
- Upload thumbnails for virality scoring
- View analytics dashboard
- Generate content ideas

---

## 🎯 Project Structure

```
youtube-viral-predictor/
├── app.py                      # Main Streamlit dashboard
├── youtube_collector.py        # YouTube data scraper
├── content_research_agent.py   # AI content generator
├── thumbnail_ctr_model.py      # CNN model trainer
├── requirements.txt            # Python dependencies
├── .env                        # API keys (DO NOT COMMIT)
├── .gitignore                  # Git ignore rules
├── data/                       # Collected datasets (gitignored)
│   ├── youtube_dataset.csv
│   └── thumbnails/
└── models/                     # Trained ML models (gitignored)
    └── thumbnail_ctr_model/
```

---

## ⚙️ Technology Stack

### Data Collection
- **YouTube Data API v3**: Video metadata and statistics
- **Google Trends (pytrends)**: Trending search queries
- **DuckDuckGo Search**: Web trend analysis

### AI & Machine Learning
- **LangChain + Groq**: AI-powered content generation using Llama 70B
- **TensorFlow/Keras**: CNN for thumbnail analysis
- **OpenCV**: Image feature extraction (face detection, color analysis)

### Web Framework
- **Streamlit**: Interactive dashboard
- **Plotly**: Data visualizations
- **Pandas**: Data processing

---

## 🚨 Important Limitations

This is a **simplified educational project** with these known limitations:

### Data Accuracy
- ❌ **Not real-time**: Data is historical and may be hours/days old
- ❌ **Limited sample size**: Based on small datasets (typically 50-200 videos)
- ❌ **Niche-specific**: Trained on tech/finance only, may not work for other niches
- ❌ **No A/B testing**: Cannot account for thumbnail variations

### Model Limitations
- ❌ **Correlation ≠ Causation**: High scores don't guarantee viral success
- ❌ **Missing context**: Doesn't account for channel authority, timing, trends, algorithm changes
- ❌ **Overfitting risk**: Small training datasets may produce unreliable predictions
- ❌ **Static features**: Only analyzes visual elements, not video content quality

### Use Cases
✅ **Good for**: Inspiration, trend analysis, learning ML/AI
❌ **Not good for**: Business decisions, guaranteed predictions, production use

---

## 📈 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub (excluding `.env`, `data/`, `models/`)
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your repository
4. Add secrets in Settings → Secrets:
```toml
YOUTUBE_API_KEY = "your_key"
GROQ_API_KEY = "your_key"
```
5. Deploy!

**Note**: Pre-train your model locally before deploying. Streamlit Cloud has limited compute resources.

---

## 🛠️ Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### API quota exceeded
- YouTube API: Wait 24 hours or use a different project
- Groq API: Implement rate limiting or wait 1 minute

### Model training fails
- Ensure you have at least 50-100 videos collected
- Check if `data/thumbnails/` folder exists with images
- Reduce model complexity if running out of memory

### Streamlit deployment issues
- Check logs in Streamlit Cloud dashboard
- Verify all secrets are added correctly
- Ensure requirements.txt has all dependencies

---

## 🤝 Contributing

This is an educational project. Contributions are welcome!

### Areas for improvement:
- [ ] Add more data sources (TikTok, Instagram)
- [ ] Improve model accuracy with larger datasets
- [ ] Add real-time data fetching
- [ ] Support more niches (gaming, lifestyle, etc.)
- [ ] Implement A/B testing framework
- [ ] Add user authentication
- [ ] Create API endpoints

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**This tool is for educational and inspirational purposes only.**

- Predictions are **not guaranteed** and should not be used as the sole basis for content decisions
- YouTube's algorithm is complex and constantly evolving
- Success depends on many factors: content quality, timing, audience, trends, SEO, etc.
- Always create authentic content that provides value to your audience
- This project does not guarantee viral success or increased views

**Use responsibly and ethically.**

---

## 🙏 Acknowledgments

- YouTube Data API for data access
- Groq for free AI inference
- Streamlit for the amazing framework
- OpenCV and TensorFlow communities

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/sujansubedi89/youtube-trend/issues)
- **Discussions**: [GitHub Discussions](https://github.com/sujansubedi89/youtube-trend/discussions)
## 🔗 Demo
Try the live demo here: (sujan-youtube-trend.streamlit.app)

## 🌟 Star History

If this project helped you, consider giving it a ⭐!

---

**Built with ❤️ for YouTube creators exploring AI and data science**

*Last updated: December 2024*

