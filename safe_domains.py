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

    # Canvas and educational platforms
    "instructure.com", "canvas.com", "blackboard.com", "moodle.org", "d2l.com",

    # Study tools
    "quizlet.com", "chegg.com", "coursehero.com", "khanacademy.org",
    "scribd.com", "slader.com", "brainly.com", "kahoot.com", "kahoot.it",
    "quizizz.com", "brilliant.org", "studystack.com",
]

safe_suffixes = [
    "edu",  # Educational institutions
    "gov",  # US Government
    "mil",  # Military
]

def safe_domain_check(url: str) -> bool:

    extracted = tldextract.extract(url)
    domain = f"{extracted.domain}.{extracted.suffix}"

    if domain in safe_domains:
        return True
    if extracted.suffix in safe_suffixes:
        return True

    return False

