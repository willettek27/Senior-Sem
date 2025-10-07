# safe_domains.py

import tldextract

safe_domains = [
    "google.com","youtube.com","facebook.com","instagram.com","x.com","reddit.com",
    "wikipedia.org","amazon.com","tiktok.com","whatsapp.com","microsoft.com",
    "github.com","stackoverflow.com","zoom.us","spotify.com","netflix.com",
    "canvas.com","instructure.com","blackboard.com","moodle.org","edu","gov","mil",
]

def safe_domain_check(url: str) -> bool:

    extracted = tldextract.extract(url)
    domain, suffix = f"{extracted.domain}.{extracted.suffix}".lower(), extracted.suffix.lower()
    
    return domain in safe_domains or suffix in safe_domains



