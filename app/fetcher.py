import praw
import logging
import json
from typing import Dict, Optional, Set
from praw.models import Submission
from prawcore.exceptions import PrawcoreException
from config.credentials import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_USERNAME,
    REDDIT_PASSWORD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SORT_TYPES = ["hot", "new", "top", "rising"]

ECOMMERCE_SUBREDDITS = [
    "ecommerce", "Entrepreneur", "smallbusiness", "shopify",
    "digitalmarketing", "affiliatemarketing", "dropship", "AmazonSeller"
]

def get_reddit_instance() -> praw.Reddit:
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        check_for_async=False
    )

def parse_post(post: Submission) -> Dict[str, Optional[str]]:
    return {
        "id": post.id,
        "title": post.title,
        "text": post.selftext,
        "url": post.url,
        "created_utc": post.created_utc,
        "score": post.score,
        "num_comments": post.num_comments,
        "author": str(post.author) if post.author else None,
        "subreddit": str(post.subreddit)
    }

def fetch_maximum_ecommerce_posts(output_file="data/ecommerce_full.json"):
    reddit = get_reddit_instance()
    seen_ids: Set[str] = set()
    all_posts = []

    for sub in ECOMMERCE_SUBREDDITS:
        subreddit = reddit.subreddit(sub)
        logger.info(f"🔍 Subreddit: r/{sub}")

        for sort in SORT_TYPES:
            try:
                fetch_method = {
                    "new": subreddit.new,
                    "hot": subreddit.hot,
                    "top": subreddit.top,
                    "rising": subreddit.rising
                }[sort]

                logger.info(f"  ↳ Fetching: {sort}")
                for post in fetch_method(limit=1000):
                    if post.id not in seen_ids:
                        parsed = parse_post(post)
                        all_posts.append(parsed)
                        seen_ids.add(post.id)

            except PrawcoreException as e:
                logger.error(f"⚠️ Error fetching from r/{sub} [{sort}]: {e}")
                continue

    logger.info(f"✅ Total posts collected: {len(all_posts)}")
    with open(output_file, "w") as f:
        json.dump(all_posts, f, indent=2)
    logger.info(f"📁 Saved to {output_file}")
