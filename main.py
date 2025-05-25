from app.fetcher import fetch_posts

if __name__ == "__main__":
    posts = fetch_posts("technology", 5)
    for post in posts:
        print(f"🧠 {post['title']}")
