import requests
import csv
import time

# Replace with your actual mediastack API key.
API_KEY = "e18cd5159f5dae9081a2066a89f30918"
BASE_URL = "http://api.mediastack.com/v1/news"

# List of keywords to search for separately.
keywords_list = [
    "Artificial General Intelligence",
    "AGI",
    "Superintelligence",
    "Machine Consciousness",
    "Strong AI"
]

# Define your date range.
from_date = "2022-01-01"
to_date   = "2024-12-31"

# Set the global maximum number of unique articles.
global_limit = 2000

# Dictionary to hold all unique articles, keyed by URL.
global_articles = {}

# Loop over each keyword until we reach the desired number of articles.
for keyword in keywords_list:
    print(f"Searching for keyword: {keyword}")
    page = 1
    while True:
        params = {
            "access_key": API_KEY,
            "keywords": keyword,
            "date": f"{from_date},{to_date}",
            "languages": "en",     # Filter for English articles.
            "limit": 100,          
            "page": page,
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        # Check for errors in the response.
        if "error" in data:
            print("Error:", data["error"])
            break
        
        results = data.get("data", [])
        if not results:
            # No more articles available for this keyword.
            break
        
        # Process the results and add new articles to the global dictionary.
        for article in results:
            url = article.get("url", "")
            if url and url not in global_articles:
                # Attach the current searched keyword for reference.
                article["searched_keyword"] = keyword
                global_articles[url] = article
            
            # Break early if we've reached the global limit.
            if len(global_articles) >= global_limit:
                break

        print(f"Keyword '{keyword}': Page {page} processed, total unique articles so far: {len(global_articles)}")
        
        if len(global_articles) >= global_limit:
            break

        # Get pagination details from the response.
        pagination = data.get("pagination", {})
        current_page = pagination.get("current", page)
        total_pages = pagination.get("pages", page)
        if current_page >= total_pages:
            break

        page += 1
        time.sleep(1)  # Pause briefly to respect rate limits.
    
    # If global limit reached, stop processing further keywords.
    if len(global_articles) >= global_limit:
        break

print(f"Total unique articles retrieved: {len(global_articles)}")

# Save the unique articles to a CSV file.
csv_filename = "mediastack_news.csv"
with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    # Define the header fields.
    fieldnames = ["searched_keyword", "title", "description", "url", "published_at"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for article in global_articles.values():
        writer.writerow({
            "searched_keyword": article.get("searched_keyword", ""),
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
        })

print(f"Articles saved to {csv_filename}")
