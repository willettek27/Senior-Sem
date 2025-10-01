# safe_domains.py

import tldextract

safe_domains = [
   # Popular websites
    "example.com", "google.com", "youtube.com", "facebook.com", "instagram.com",
    "chatgpt.com", "x.com", "reddit.com", "wikipedia.org", "amazon.com",
    "tiktok.com", "whatsapp.com", "canva.com", "booking.com", "ebay.com",
    "duckduckgo.com", "yahoo.com", "telegram.org", "roblox.com", "samsung.com",
    "mail.ru", "globo.com", "sharepoint.com", "live.com", "office.com",
    "netflix.com", "linkedin.com", "microsoft.com", "zoom.us", "pinterest.com",
    "wordpress.com", "adobe.com", "github.com", "stackoverflow.com", "imdb.com",
    "etsy.com", "bbc.com", "cnn.com", "nytimes.com", "hulu.com", "spotify.com",
    "dropbox.com",

    # Educational and governmental sites
        "harvard.edu", "stanford.edu", "mit.edu", "yale.edu", "princeton.edu",
        "rockford.edu", "rockvalleycollege.edu"
    
    #Canvas
    "instructure.com"

]

def safe_domain_check(url: str) -> bool:

    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"
    return domain in safe_domains

