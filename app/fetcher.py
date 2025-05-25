import praw
from config.credentials import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_USERNAME,
    REDDIT_PASSWORD
)

def get_reddit_instance():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        check_for_async=False
    )
    return reddit

def fetch_posts(subreddit_name="technology", limit=10):
    reddit = get_reddit_instance()
    posts = []
    for post in reddit.subreddit(subreddit_name).new(limit=limit):
        posts.append({
            "title": post.title,
            "text": post.selftext,
            "created_utc": post.created_utc
        })
    return posts
