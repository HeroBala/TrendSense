# ========== LIBRARY IMPORTS ==========

import praw  # Main library for interacting with Reddit's API.
import logging  # Standard Python library for application logging.
import json  # For reading and writing JSON data to files.
import time  # Provides time-related functions (delays, timing execution).
import socket  # Used to set network-level timeouts for API calls.
from typing import List, Dict, Optional, Set  # Used for static type checking.
from praw.models import Submission, Comment  # Datatypes for Reddit posts and comments.
from prawcore.exceptions import PrawcoreException, ResponseException, RequestException  # Error classes for robust API error handling.
from config.credentials import (  # Importing all credentials from a separate config module for security.
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_USERNAME,
    REDDIT_PASSWORD
)

# ========== CONFIGURATION SECTION ==========

# Set the default timeout for all socket operations (in seconds).
socket.setdefaulttimeout(30)

# List of sort types to use for Reddit post fetching. Each one returns posts in a different order.
SORT_TYPES = ["hot", "new", "top", "rising"]

# Maximum number of retry attempts for network or API failures.
MAX_RETRIES = 3

# Minimum upvote score that a post must have to be included in output.
MIN_SCORE = 5

# Maximum number of posts to fetch for each sort type per subreddit.
MAX_POSTS_PER_SORT = 300

# Maximum number of top-level comments to fetch for each Reddit post.
MAX_COMMENTS_PER_POST = 10

# Path to the output file that will store all fetched data in JSON format.
OUTPUT_FILE = "data/ecommerce_advanced.json"

# List of subreddit names to fetch posts from. Can be modified to target specific Reddit communities.
SUBREDDITS = [
    "ecommerce", "Entrepreneur", "smallbusiness", "shopify", "WooCommerce", "bigcommerce",
    "digitalmarketing", "affiliatemarketing", "dropship", "AmazonSeller", "FulfillmentByAmazon",
    "printondemand", "EcommerceSEO", "EtsySellers", "JustStart", "FBA", "AmazonFBA", "OnlineRetail",
    "eBay", "SEO", "content_marketing", "PPC", "sidehustle", "growthhacking", "Startup_Ideas",
    "Startups", "Passive_Income", "marketing", "emailmarketing", "dataisbeautiful", "UXDesign",
    "userexperience", "ShopifyHelp", "shopifyapps", "etsy", "dropshipping", "shopifydev",
    "branding", "Startup", "webmarketing", "onlinemarketing", "SocialMediaMarketing"
]

# ========== LOGGING SETUP ==========

# Set up logging to report info, warnings, and errors with timestamps and log levels.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========== REDDIT AUTHENTICATION ==========

def get_reddit_instance() -> praw.Reddit:
    """
    Creates and returns an authenticated Reddit API instance using credentials.
    Raises an exception if credentials are incorrect or Reddit is unreachable.
    """
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        check_for_async=False  # Ensures all actions are performed synchronously.
    )
    # Test authentication; will raise if credentials are bad.
    reddit.user.me()
    logger.info("🔐 Authenticated with Reddit")
    return reddit

# ========== POST PARSING ==========

def parse_post(post: Submission, sort_type: str) -> Optional[Dict]:
    """
    Extracts all relevant details from a Reddit post, including top comments.
    Filters out comments that are blank, deleted, or not of type Comment.
    Returns a dictionary suitable for saving or further processing, or None if parsing fails.
    """
    try:
        # Ensures all comments are loaded and not collapsed into 'MoreComments' objects.
        post.comments.replace_more(limit=0)
        # Collects up to MAX_COMMENTS_PER_POST top-level comments (not replies).
        comments = [
            c.body.strip()
            for c in post.comments[:MAX_COMMENTS_PER_POST]
            if isinstance(c, Comment) and c.body.strip()
        ]
        return {
            "id": post.id,
            "title": post.title.strip(),
            "text": post.selftext.strip(),  # Main text of the post; can be empty for link/image posts.
            "url": post.url,
            "created_utc": post.created_utc,
            "score": post.score,
            "num_comments": post.num_comments,
            "author": str(post.author) if post.author else None,  # Handles deleted or suspended users.
            "subreddit": str(post.subreddit),
            "sort": sort_type,  # Records which sort mode was used to find the post.
            "comments": comments
        }
    except Exception as e:
        # If parsing fails (e.g., due to removed/deleted content), log a warning and skip this post.
        logger.warning(f"⚠️ Skipping post {post.id}: {e}")
        return None

# ========== SUBREDDIT FETCHING ==========

def fetch_subreddit(reddit: praw.Reddit, name: str, seen_ids: Set[str]) -> List[Dict]:
    """
    Downloads posts from a single subreddit, using all specified sort types.
    Avoids processing duplicate posts by tracking post IDs in seen_ids.
    Retries failed requests up to MAX_RETRIES times, with a delay between retries.
    Returns a list of parsed post dictionaries for this subreddit.
    """
    subreddit = reddit.subreddit(name)
    results = []

    for sort in SORT_TYPES:
        logger.info(f"📥 Fetching r/{name} [{sort}]")
        for attempt in range(MAX_RETRIES):
            try:
                # Fetch posts using the current sort type (e.g., hot, new).
                fetch = getattr(subreddit, sort)(limit=MAX_POSTS_PER_SORT)
                for post in fetch:
                    # Skip posts already seen or not meeting the minimum score.
                    if post.id in seen_ids or post.score < MIN_SCORE:
                        continue
                    parsed = parse_post(post, sort)
                    if parsed:
                        results.append(parsed)
                        seen_ids.add(post.id)  # Mark this post as seen to avoid duplicates.
                break  # Successful fetch, exit retry loop for this sort.
            except (PrawcoreException, RequestException, ResponseException, socket.timeout) as e:
                # Log API/network errors and retry after a short pause.
                logger.warning(f"⏳ Retry {attempt+1}/{MAX_RETRIES} - r/{name} [{sort}] failed: {e}")
                time.sleep(5)
        else:
            # If all retries failed for this sort, log an error and skip.
            logger.error(f"❌ Skipped r/{name} [{sort}] after {MAX_RETRIES} retries.")

    return results

# ========== MAIN DATA PIPELINE ==========

def fetch_all_data():
    """
    Main process to fetch posts across all subreddits in SUBREDDITS.
    Handles authentication, duplicate post filtering, and progress logging.
    Writes all collected post data to OUTPUT_FILE in JSON format.
    Also logs total time taken and number of posts collected.
    """
    start = time.time()
    reddit = get_reddit_instance()
    all_data: List[Dict] = []
    seen_ids: Set[str] = set()  # Used to avoid collecting duplicate posts.

    for idx, subreddit in enumerate(SUBREDDITS):
        posts = fetch_subreddit(reddit, subreddit, seen_ids)
        all_data.extend(posts)
        logger.info(f"✅ {len(posts)} posts from r/{subreddit} ({idx+1}/{len(SUBREDDITS)})")

    # Save all collected post data to a JSON file with readable formatting.
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    duration = time.time() - start
    logger.info(f"📊 Done. {len(all_data)} total posts saved to {OUTPUT_FILE}")
    logger.info(f"⏱️ Time taken: {duration:.2f} seconds")

# ========== SCRIPT ENTRY POINT ==========

if __name__ == "__main__":
    # Only runs if this script is run directly (not imported as a module).
    fetch_all_data()
