import os
import json
import requests
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pytrends.request import TrendReq

class ContentResearchAgent:
    def __init__(self, groq_api_key):
        # Free Groq LLM (llama3-70b is free!)
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",  # Free tier
            temperature=0.7
        )
        
        # Google Trends (free, no API key!)
        self.pytrends = TrendReq(hl='en-US', tz=360)
        
        # DuckDuckGo search - direct implementation
        try:
            from duckduckgo_search import DDGS
            self.ddgs = DDGS()
            self.search_available = True
        except:
            print("⚠️ DuckDuckGo search not available, continuing without it")
            self.ddgs = None
            self.search_available = False
    
    
    def research_google_trends(self, niche='tech'):
        """Get trending topics from Google Trends (FREE!)"""
        try:
            keywords_map = {
                'tech': ['AI', 'Python', 'Machine Learning', 'Web Development', 'Coding'],
                'finance': ['Investing', 'Stocks', 'Crypto', 'Personal Finance', 'Trading']
            }
            
            keywords = keywords_map.get(niche, keywords_map['tech'])
            
            # Get trending searches
            self.pytrends.build_payload(keywords, timeframe='now 7-d')
            interest = self.pytrends.interest_over_time()
            
            # Get related queries
            related = self.pytrends.related_queries()
            
            trending_topics = []
            for keyword in keywords:
                if keyword in related and related[keyword]['top'] is not None:
                    top_queries = related[keyword]['top'].head(5)
                    for idx, row in top_queries.iterrows():
                        trending_topics.append({
                            'query': row['query'],
                            'value': row['value'],
                            'category': keyword,
                            'source': 'Google Trends'
                        })
            
            print(f"✅ Found {len(trending_topics)} trending topics from Google Trends")
            return trending_topics
            
        except Exception as e:
            print(f"⚠️ Google Trends error: {e}")
            return []
    
    def search_twitter_trends(self, niche='tech'):
        """Search Twitter/X trends using web scraping (FREE!)"""
        if not self.search_available:
            print("⚠️ Web search unavailable, skipping Twitter search")
            return "No Twitter data available"
        
        try:
            # Use DuckDuckGo to find recent Twitter discussions
            queries = {
                'tech': 'site:twitter.com (AI OR programming OR tech)',
                'finance': 'site:twitter.com (investing OR stocks OR finance)'
            }
            
            query = queries.get(niche, queries['tech'])
            results = self.ddgs.text(query, max_results=5)
            
            result_list = []
            for r in results:
                result_list.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', '')
                })
            
            print(f"✅ Found Twitter trends via web search")
            return result_list
            
        except Exception as e:
            print(f"⚠️ Twitter search error: {e}")
            return "No Twitter data available"
    
    def search_web_trends(self, niche='tech'):
        """Search web for trending topics using DuckDuckGo (FREE!)"""
        if not self.search_available:
            print("⚠️ Web search unavailable, skipping")
            return []
        
        queries = {
            'tech': [
                'latest AI technology trends 2024',
                'trending programming topics',
                'hot tech news today',
                'viral coding tutorials'
            ],
            'finance': [
                'trending investment topics 2024',
                'hot stocks today',
                'crypto trends',
                'personal finance tips trending'
            ]
        }
        
        all_results = []
        for query in queries.get(niche, queries['tech']):
            try:
                results = self.ddgs.text(query, max_results=5)
                result_list = []
                for r in results:
                    result_list.append({
                        'title': r.get('title', ''),
                        'snippet': r.get('body', ''),
                        'url': r.get('href', '')
                    })
                all_results.append({
                    'query': query,
                    'results': result_list
                })
                print(f"✅ Searched: {query}")
            except Exception as e:
                print(f"⚠️ Search error for '{query}': {e}")
        
        return all_results
    
    def analyze_audience_sentiment(self, topics_data, niche):
        """Analyze what audience cares about using AI"""
        prompt_template = """
You are a YouTube content strategist analyzing trending topics.

Niche: {niche}

Recent trending data from multiple sources:
{topics}

Based on this data, identify:
1. The top 3 themes people are most interested in
2. Common pain points or questions
3. Emotional triggers (curiosity, fear, excitement)
4. Content gaps or underserved topics

Provide a concise analysis in JSON format:
{{
  "top_themes": ["theme1", "theme2", "theme3"],
  "pain_points": ["point1", "point2"],
  "emotional_triggers": ["trigger1", "trigger2"],
  "content_gaps": ["gap1", "gap2"]
}}
"""
        
        # Format topics from various sources
        topics_str = ""
        
        # Add Google Trends data
        if 'google_trends' in topics_data:
            topics_str += "\n\nGoogle Trends:\n"
            for t in topics_data['google_trends'][:10]:
                topics_str += f"- {t.get('query', '')} (Interest: {t.get('value', 'N/A')})\n"
        
        # Add web search results
        if 'web_search' in topics_data:
            topics_str += "\n\nWeb Trends:\n"
            topics_str += str(topics_data['web_search'][:500])  # Truncate for token limit
        
        try:
            # Format prompt
            formatted_prompt = prompt_template.format(topics=topics_str, niche=niche)
            
            # Use LLM directly
            response = self.llm.invoke(formatted_prompt)
            response_text = response.content
            
            # Extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            return json.loads(response_text[start:end])
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return {
                "top_themes": ["AI and automation", "Career development", "Tutorials"],
                "pain_points": ["Getting started", "Best practices"],
                "emotional_triggers": ["FOMO", "Curiosity"],
                "content_gaps": ["Beginner-friendly content"]
            }
    
    def generate_script_outlines(self, sentiment_data, niche, num_outlines=5):
        """Generate video script outlines using AI"""
        prompt_template = """
You are an expert YouTube content creator in the {niche} niche.

Audience Analysis:
{sentiment}

Generate {num} viral video script outlines that will perform well based on this data.

For each outline, provide:
1. **Catchy Title** (50-70 characters, includes power words)
2. **Hook** (First 10 seconds to grab attention)
3. **Main Points** (3-4 key sections)
4. **Call-to-Action**
5. **Estimated Watch Time** (8-15 minutes ideal)
6. **Virality Potential** (score 1-10 with reasoning)

Format each outline clearly with markdown headers.
"""
        
        sentiment_str = json.dumps(sentiment_data, indent=2)
        
        try:
            # Format prompt
            formatted_prompt = prompt_template.format(
                sentiment=sentiment_str,
                niche=niche,
                num=num_outlines
            )
            
            # Use LLM directly
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            print(f"Error generating outlines: {e}")
            return "Error generating content"
    
    def run_full_research(self, niche='tech'):
        """Complete research pipeline (Reddit-free!)"""
        print(f"🔍 Starting content research for '{niche}' niche...\n")
        
        all_topics = {}
        
        # Step 1: Google Trends
        print("📊 Analyzing Google Trends...")
        google_trends = self.research_google_trends(niche)
        all_topics['google_trends'] = google_trends
        if google_trends:
            print(f"   Found {len(google_trends)} trending topics\n")
        
        # Step 2: Web search (DuckDuckGo)
        print("🌐 Searching web trends...")
        web_trends = self.search_web_trends(niche)
        all_topics['web_search'] = web_trends
        print(f"   Collected web data\n")
        
        # Step 3: Twitter trends
        print("🐦 Searching Twitter trends...")
        twitter_trends = self.search_twitter_trends(niche)
        all_topics['twitter'] = twitter_trends
        print(f"   Collected Twitter data\n")
        
        # Step 4: Sentiment analysis
        print("🧠 Analyzing audience sentiment...")
        sentiment = self.analyze_audience_sentiment(all_topics, niche)
        print(f"   Top themes: {', '.join(sentiment['top_themes'])}\n")
        
        # Step 5: Generate outlines
        print("✍️ Generating video script outlines...")
        outlines = self.generate_script_outlines(sentiment, niche, num_outlines=5)
        
        # Save results
        results = {
            'niche': niche,
            'timestamp': datetime.now().isoformat(),
            'google_trends': google_trends,
            'web_trends': web_trends[:5] if web_trends else [],  # Limit size
            'twitter_trends': str(twitter_trends)[:500],  # Truncate
            'sentiment_analysis': sentiment,
            'script_outlines': outlines
        }
        
        os.makedirs('data', exist_ok=True)
        with open(f'data/content_research_{niche}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n✅ Research complete!")
        print(f"📁 Results saved to data/content_research_{niche}.json\n")
        
        return outlines


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load API keys from .env file
    load_dotenv()
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    if not GROQ_API_KEY:
        print("❌ Error: GROQ_API_KEY not found in .env file!")
        print("Get a free key from: https://console.groq.com/")
        print("Then add to .env: GROQ_API_KEY=your_key_here")
        exit(1)
    
    agent = ContentResearchAgent(groq_api_key=GROQ_API_KEY)
    
    # Run research for tech niche (NO REDDIT NEEDED!)
    outlines = agent.run_full_research(niche='tech')
    print("\n" + "="*60)
    print("GENERATED SCRIPT OUTLINES:")
    print("="*60)
    print(outlines)