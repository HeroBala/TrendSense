
import praw
import logging
import json
import time
import socket
from typing import List, Dict, Optional, Set
from praw.models import Submission, Comment
from prawcore.exceptions import PrawcoreException, ResponseException, RequestException
from config.credentials import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_USERNAME,
    REDDIT_PASSWORD
)

# ========== CONFIGURATION ==========
socket.setdefaulttimeout(30)
SORT_TYPES = ["hot", "new", "top", "rising"]
MAX_RETRIES = 3
MIN_SCORE = 5
MAX_POSTS_PER_SORT = 300
MAX_COMMENTS_PER_POST = 10
OUTPUT_FILE = "data/ecommerce_advanced.json"

SUBREDDITS = [
    "ecommerce", "Entrepreneur", "smallbusiness", "shopify", "WooCommerce", "bigcommerce",
    "digitalmarketing", "affiliatemarketing", "dropship", "AmazonSeller", "FulfillmentByAmazon",
    "printondemand", "EcommerceSEO", "EtsySellers", "JustStart", "FBA", "AmazonFBA", "OnlineRetail",
    "eBay", "SEO", "content_marketing", "PPC", "sidehustle", "growthhacking", "Startup_Ideas",
    "Startups", "Passive_Income", "marketing", "emailmarketing", "dataisbeautiful", "UXDesign",
    "userexperience", "ShopifyHelp", "shopifyapps", "etsy", "dropshipping", "shopifydev",
    "branding", "Startup", "webmarketing", "onlinemarketing", "SocialMediaMarketing"
]

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ========== REDDIT INSTANCE ==========
def get_reddit_instance() -> praw.Reddit:
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        check_for_async=False
    )
    reddit.user.me()
    logger.info("🔐 Authenticated with Reddit")
    return reddit


# ========== POST PARSER ==========
def parse_post(post: Submission, sort_type: str) -> Optional[Dict]:
    try:
        post.comments.replace_more(limit=0)
        comments = [
            c.body.strip()
            for c in post.comments[:MAX_COMMENTS_PER_POST]
            if isinstance(c, Comment) and c.body.strip()
        ]
        return {
            "id": post.id,
            "title": post.title.strip(),
            "text": post.selftext.strip(),
            "url": post.url,
            "created_utc": post.created_utc,
            "score": post.score,
            "num_comments": post.num_comments,
            "author": str(post.author) if post.author else None,
            "subreddit": str(post.subreddit),
            "sort": sort_type,
            "comments": comments
        }
    except Exception as e:
        logger.warning(f"⚠️ Skipping post {post.id}: {e}")
        return None


# ========== FETCH POSTS ==========
def fetch_subreddit(reddit: praw.Reddit, name: str, seen_ids: Set[str]) -> List[Dict]:
    subreddit = reddit.subreddit(name)
    results = []

    for sort in SORT_TYPES:
        logger.info(f"📥 Fetching r/{name} [{sort}]")
        for attempt in range(MAX_RETRIES):
            try:
                fetch = getattr(subreddit, sort)(limit=MAX_POSTS_PER_SORT)
                for post in fetch:
                    if post.id in seen_ids or post.score < MIN_SCORE:
                        continue
                    parsed = parse_post(post, sort)
                    if parsed:
                        results.append(parsed)
                        seen_ids.add(post.id)
                break  # Success
            except (PrawcoreException, RequestException, ResponseException, socket.timeout) as e:
                logger.warning(f"⏳ Retry {attempt+1}/{MAX_RETRIES} - r/{name} [{sort}] failed: {e}")
                time.sleep(5)
        else:
            logger.error(f"❌ Skipped r/{name} [{sort}] after {MAX_RETRIES} retries.")

    return results


# ========== MAIN PIPELINE ==========
def fetch_all_data():
    start = time.time()
    reddit = get_reddit_instance()
    all_data: List[Dict] = []
    seen_ids: Set[str] = set()

    for idx, subreddit in enumerate(SUBREDDITS):
        posts = fetch_subreddit(reddit, subreddit, seen_ids)
        all_data.extend(posts)
        logger.info(f"✅ {len(posts)} posts from r/{subreddit} ({idx+1}/{len(SUBREDDITS)})")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    duration = time.time() - start
    logger.info(f"📊 Done. {len(all_data)} total posts saved to {OUTPUT_FILE}")
    logger.info(f"⏱️ Time taken: {duration:.2f} seconds")


# ========== RUN ==========
if __name__ == "__main__":
    fetch_all_data()
