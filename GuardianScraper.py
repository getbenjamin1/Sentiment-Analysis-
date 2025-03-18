import requests
import csv
import time

# Replace with your actual Guardian API key.
API_KEY = '8e54b13a-2c29-4cd4-9fe7-67e3019bbfff'
BASE_URL = 'https://content.guardianapis.com/search'

# Define your search parameters.
keywords = "Artificial General Intelligence OR AGI OR Superintelligence OR Machine Consciousness OR Strong AI"
from_date = "2022-01-01"
to_date = "2024-12-31"

# Set the initial page and create an empty list to store articles.
page = 1
articles = []

# Specify that you only need the trailText field (i.e. a summary).
show_fields = "trailText"

# Define the maximum number of articles to retrieve.
max_articles = 2000

while True:
    params = {
        'q': keywords,
        'from-date': from_date,
        'to-date': to_date,
        'page-size': 50,  # Maximum allowed for the free tier is typically 50
        'page': page,
        'api-key': API_KEY,
        'show-fields': show_fields  # Request only the trailText for each article
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    # Check for a valid response.
    if data.get("response", {}).get("status") != "ok":
        print("Error in API response:", data)
        break

    results = data["response"]["results"]
    if not results:
        break  # No more articles to process.
    
    articles.extend(results)
    print(f"Page {page} retrieved, total articles so far: {len(articles)}")
    
    # Check if we've reached the maximum number of articles.
    if len(articles) >= max_articles:
        # Trim the list to exactly max_articles if necessary.
        articles = articles[:max_articles]
        break

    # Pagination: stop if this is the last page.
    current_page = data["response"]["currentPage"]
    total_pages = data["response"]["pages"]
    if current_page >= total_pages:
        break

    page += 1
    time.sleep(1)  # Pause briefly to respect rate limits.

print(f"Total articles retrieved: {len(articles)}")

# Save the articles to a CSV file.
csv_filename = "articles.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    # Define the header fields you want to save.
    fieldnames = [
        "id", 
        "webPublicationDate", 
        "webTitle", 
        "webUrl", 
        "sectionName", 
        "trailText"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for article in articles:
        fields = article.get("fields", {})
        writer.writerow({
            "id": article.get("id"),
            "webPublicationDate": article.get("webPublicationDate"),
            "webTitle": article.get("webTitle"),
            "webUrl": article.get("webUrl"),
            "sectionName": article.get("sectionName"),
            "trailText": fields.get("trailText")
        })

print(f"Articles saved to {csv_filename}")
