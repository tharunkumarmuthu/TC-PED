import asyncio
import pandas as pd
import os
from twikit import Client

# --- Configuration ---

COOKIE_FILE = 'twikit_cookies.json'

# Optimized TN Politics Query
QUERY = '("Tamil Nadu" OR TamilNadu OR தமிழ்நாடு OR தமிழகம் OR #TamilNadu OR #TNPolitics OR #TNPolls OR DMK OR AIADMK OR திமுக OR அதிமுக OR "நாம் தமிழர்" OR பாஜக OR TVK OR தவெக OR #DMK OR #AIADMK OR Stalin OR Annamalai OR EPS OR Vijay OR விஜய்) (lang:ta OR lang:en) min_faves:50 min_retweets:20 filter:has_engagement -filter:replies'

async def get_client():
    """Initializes client and handles cookie loading/logging in."""
    client = Client('en-US')
    
    if os.path.exists(COOKIE_FILE):
        # Load existing session to avoid re-login
        client.load_cookies(COOKIE_FILE)
        print("✅ Session loaded from cookies. No login required.")
    else:
        # Perform fresh login and save cookies
        print("🔑 No cookies found")
        exit()
    
    return client

async def scrape_tn_data():
    client = await get_client()
    tweets_data = []
    
    print(f"🔍 Searching for: {QUERY}")
    # product='Top' ensures you get the most relevant political discourse
    tweets = await client.search_tweet(QUERY, product='Top')

    count = 0
    max_tweets = 100 

    while count < max_tweets:
        for tweet in tweets:
            user = tweet.user
            # Get deep user details
            user = tweet.user
            
            tweets_data.append({
                # --- Tweet Core Details ---
                'tweet_id': tweet.id,
                'created_at': tweet.created_at,
                'full_text': tweet.full_text,
                'lang': tweet.lang,
                'is_quote': tweet.is_quote_status,
                'possibly_sensitive': getattr(tweet, 'possibly_sensitive', False),
                
                # --- Metrics (Vital for Weighting in Models) ---
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count,
                'reply_count': tweet.reply_count,
                'quote_count': tweet.quote_count,
                'view_count': getattr(tweet, 'view_count', 'N/A'),
                
                # --- Deep User Details ---
                'user_id': user.id,
                'user_name': user.name,
                'screen_name': user.screen_name,
                'user_desc': user.description,
                'user_location': user.location,
                'user_verified': user.verified,
                'is_blue_verified': getattr(user, 'is_blue_verified', False),
                'followers_count': user.followers_count,
                'following_count': user.following_count,
                'total_tweets_by_user': user.statuses_count,
                
                # --- Entities & Geo ---
                'hashtags': tweet.hashtags,
                'urls': [u['expanded_url'] for u in getattr(tweet, 'urls', [])],
                'place': tweet.place.full_name if tweet.place else 'N/A',
            })

            count += 1

        # Pagination logic
        try:
            print(f"📦 Progress: {len(tweets_data)} tweets collected...")
            tweets = await tweets.next()
            await asyncio.sleep(5) # Crucial: prevents rate limiting (429 errors)
        except Exception as e:
            print(f"⚠️ Reached end of results or rate limit: {e}")
            break

    # Save to CSV
    if tweets_data:
        df = pd.DataFrame(tweets_data)
        # utf-8-sig ensures Tamil script renders correctly in Excel/CSV
        df.to_csv('tn_political_data_full_2.csv', index=False, encoding='utf-8-sig')
        print(f"🚀 Success! {len(df)} tweets saved to 'tn_political_data_full_2.csv'")
    else:
        print("❌ No data collected.")

if __name__ == "__main__":
    asyncio.run(scrape_tn_data())