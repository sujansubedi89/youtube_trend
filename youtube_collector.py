import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from PIL import Image
from io import BytesIO

class YouTubeDataCollector:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.collected_data = []
    
    def search_videos(self, query, niche='tech', max_results=50, days_back=30):
        """Search for videos in a specific niche"""
        published_after = (datetime.now() - timedelta(days=days_back)).isoformat() + 'Z'
        
        # Niche-specific queries
        niche_queries = {
            'tech': [f'{query} tech', f'{query} programming', f'{query} AI', 
                    f'{query} software', f'{query} coding'],
            'finance': [f'{query} finance', f'{query} investing', f'{query} stocks',
                       f'{query} crypto', f'{query} money']
        }
        
        all_videos = []
        
        for search_query in niche_queries.get(niche, [query]):
            try:
                request = self.youtube.search().list(
                    part='id,snippet',
                    q=search_query,
                    type='video',
                    order='viewCount',
                    publishedAfter=published_after,
                    maxResults=max_results,
                    relevanceLanguage='en'
                )
                response = request.execute()
                
                for item in response.get('items', []):
                    video_id = item['id']['videoId']
                    video_data = self.get_video_details(video_id)
                    if video_data:
                        all_videos.append(video_data)
                        
            except Exception as e:
                print(f"Error searching for '{search_query}': {e}")
        
        return all_videos
    
    def get_video_details(self, video_id):
        """Get detailed statistics for a video"""
        try:
            request = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                return None
            
            item = response['items'][0]
            snippet = item['snippet']
            stats = item['statistics']
            
            # Download thumbnail
            thumbnail_url = snippet['thumbnails']['high']['url']
            thumbnail_data = self.download_thumbnail(thumbnail_url, video_id)
            
            # Safely get statistics with defaults
            view_count = int(stats.get('viewCount', 0))
            like_count = int(stats.get('likeCount', 0))
            comment_count = int(stats.get('commentCount', 0))
            
            return {
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet['description'][:500],
                'channel_title': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'thumbnail_url': thumbnail_url,
                'thumbnail_path': thumbnail_data,
                'view_count': view_count,
                'like_count': like_count,
                'comment_count': comment_count,
                'duration': item['contentDetails']['duration'],
                'tags': snippet.get('tags', [])
            }
        except Exception as e:
            print(f"Error getting video {video_id}: {e}")
            return None
    
    def download_thumbnail(self, url, video_id):
        """Download and save thumbnail"""
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            
            # Create thumbnails directory
            os.makedirs('data/thumbnails', exist_ok=True)
            filepath = f'data/thumbnails/{video_id}.jpg'
            img.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error downloading thumbnail: {e}")
            return None
    
    def calculate_engagement_rate(self, video_data):
        """Calculate engagement metrics"""
        views = video_data['view_count']
        likes = video_data['like_count']
        comments = video_data['comment_count']
        
        if views == 0:
            return 0
        
        engagement = ((likes + comments * 2) / views) * 100
        return round(engagement, 4)
    
    def collect_dataset(self, queries, niche='tech', max_results=50):
        """Collect complete dataset"""
        all_data = []
        
        for query in queries:
            print(f"Collecting data for: {query}")
            videos = self.search_videos(query, niche, max_results)
            all_data.extend(videos)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            print("❌ No data collected!")
            return df
        
        # Convert to numeric (fix string/int issues)
        df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
        df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0)
        df['comment_count'] = pd.to_numeric(df['comment_count'], errors='coerce').fillna(0)
        
        # Add engagement rate
        df['engagement_rate'] = df.apply(self.calculate_engagement_rate, axis=1)
        
        # Calculate virality score (normalized) - with safety checks
        max_views = df['view_count'].max()
        max_engagement = df['engagement_rate'].max()
        max_likes = df['like_count'].max()
        
        # Avoid division by zero
        if max_views > 0 and max_engagement > 0 and max_likes > 0:
            df['virality_score'] = (
                (df['view_count'] / max_views * 0.5) +
                (df['engagement_rate'] / max_engagement * 0.3) +
                (df['like_count'] / max_likes * 0.2)
            ) * 100
        else:
            df['virality_score'] = 0
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/youtube_dataset.csv', index=False)
        
        print(f"\n✅ Collected {len(df)} videos")
        print(f"Average views: {df['view_count'].mean():.0f}")
        print(f"Top video: {df.loc[df['view_count'].idxmax(), 'title']}")
        
        return df


# Example usage
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load API key from .env file
    load_dotenv()
    API_KEY = os.getenv('YOUTUBE_API_KEY')
    
    if not API_KEY:
        print("❌ Error: YOUTUBE_API_KEY not found in .env file!")
        print("Create a .env file with: YOUTUBE_API_KEY=your_key_here")
        exit(1)
    
    collector = YouTubeDataCollector(API_KEY)
    
    # Tech niche queries
    tech_queries = [
        'AI tutorial', 'machine learning', 'python programming',
        'web development', 'coding interview', 'tech news'
    ]
    
    # Finance niche queries
    finance_queries = [
        'stock analysis', 'crypto trading', 'investing tips',
        'financial freedom', 'passive income', 'market analysis'
    ]
    
    # Collect data (choose your niche)
    df = collector.collect_dataset(tech_queries, niche='tech', max_results=30)
    # df = collector.collect_dataset(finance_queries, niche='finance', max_results=30)