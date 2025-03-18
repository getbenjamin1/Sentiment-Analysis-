

import praw
import csv
import time

reddit = praw.Reddit(
    client_id='Wq5ojsd3mDbNDqyQrM-UhQ',
    client_secret='UoPwglVWBv2BVwtfJru4NcTz4qu5rg',
    user_agent='script:Dennis4th:1.0 (by /u/Zealousideal-Gain753)',
    username='Zealousideal-Gain753',
    
)

# Test the connection
print(f"Read-only mode: {reddit.read_only}")  # Should print True if authentication is successful

# Keywords to search for
search_terms = ['Artificial General Intelligence', 'AGI', 'Superintelligence', 'Machine Consciousness', 'Strong AI']

# CSV File Setup
output_file = 'reddit_data.csv'
fields = ['Keyword', 'Subreddit', 'Post Title', 'Post Content', 'Upvotes', 'Comments', 'Post URL']
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(fields)  # Write the header row

    # Fetch and save posts
    for term in search_terms:
        print(f"\nFetching posts for keyword: {term}")
        count = 0
        after = None  # Used for pagination
        max_posts = 1000  # Desired number of posts per keyword
        
        while count < max_posts:
            try:
                # Fetch posts using 'after' for pagination
                for submission in reddit.subreddit('all').search(term, limit=100, params={'after': after}):
                    count += 1
                    post_content = submission.selftext if submission.selftext else "No content (Link Post)"
                    writer.writerow([
                        term,
                        submission.subreddit.display_name,
                        submission.title,
                        post_content,
                        submission.score,
                        submission.num_comments,
                        submission.url
                    ])
                    # Update 'after' with the ID of the last fetched submission
                    after = submission.id

                    # Stop if we've reached the maximum desired posts
                    if count >= max_posts:
                        break
                
                print(f"Total posts retrieved for '{term}': {count}")

                # Sleep to avoid hitting the rate limit
                time.sleep(2)
            
            except Exception as e:
                print(f"Error while fetching posts for keyword '{term}': {e}")
                time.sleep(10)  # Wait before retrying

print(f"\nData collection complete. Saved to {output_file}.")
