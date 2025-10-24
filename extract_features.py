# =========================================
# extract_features.py
# Extract 89 features from URL + content
# =========================================

import re
import socket
import tldextract
import requests
import whois
import datetime
import urllib.parse as urlparse
from bs4 import BeautifulSoup

FEATURE_ORDER = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm",
    "nb_and", "nb_or", "nb_eq", "nb_underscore", "nb_tilde", "nb_percent", "nb_slash",
    "nb_star", "nb_colon", "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space", "nb_www",
    "nb_com", "nb_dslash", "http_in_path", "https_token", "ratio_digits_url",
    "ratio_digits_host", "punycode", "port", "tld_in_path", "tld_in_subdomain",
    "abnormal_subdomain", "nb_subdomains", "prefix_suffix", "random_domain",
    "shortening_service", "path_extension", "length_words_raw", "char_repeat",
    "shortest_words_raw", "shortest_word_host", "shortest_word_path", "longest_words_raw",
    "longest_word_host", "longest_word_path", "avg_words_raw", "avg_word_host",
    "avg_word_path", "phish_hints", "domain_in_brand", "brand_in_subdomain",
    "brand_in_path", "suspecious_tld", "statistical_report", "nb_hyperlinks",
    "ratio_intHyperlinks", "ratio_extHyperlinks", "ratio_nullHyperlinks", "nb_extCSS",
    "ratio_intRedirection", "ratio_extRedirection", "ratio_intErrors", "ratio_extErrors",
    "login_form", "external_favicon", "links_in_tags", "submit_email", "ratio_intMedia",
    "ratio_extMedia", "sfh", "iframe", "popup_window", "safe_anchor", "onmouseover",
    "right_clic", "empty_title", "domain_in_title", "domain_with_copyright",
    "whois_registered_domain", "domain_registration_length", "domain_age", "web_traffic",
    "dns_record", "google_index", "page_rank", "status"
]

def extract_features(sample):
    url = sample.get("url", "")
    content = sample.get("content", "")
    features = {}

    parsed = urlparse.urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # --- URL BASIC INFO ---
    features["length_url"] = len(url)
    features["length_hostname"] = len(hostname)
    features["ip"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0

    # Character counts
    char_map = {
        "nb_dots": r"\.", "nb_hyphens": "-", "nb_at": "@", "nb_qm": r"\?",
        "nb_and": "&", "nb_or": r"\|", "nb_eq": "=", "nb_underscore": "_",
        "nb_tilde": "~", "nb_percent": "%", "nb_slash": "/", "nb_star": r"\*",
        "nb_colon": ":", "nb_comma": ",", "nb_semicolumn": ";", "nb_dollar": r"\$",
        "nb_space": " "
    }
    for k, v in char_map.items():
        features[k] = len(re.findall(v, url))

    features["nb_www"] = url.count("www")
    features["nb_com"] = url.count(".com")
    features["nb_dslash"] = url.count("//")
    features["http_in_path"] = 1 if "http" in path else 0
    features["https_token"] = 1 if "https" in hostname or "https" in path else 0

    digits = len(re.findall(r"\d", url))
    features["ratio_digits_url"] = digits / len(url) if len(url) else 0
    features["ratio_digits_host"] = sum(c.isdigit() for c in hostname) / len(hostname) if hostname else 0
    features["punycode"] = 1 if "xn--" in hostname else 0
    features["port"] = 1 if parsed.port else 0

    ext = tldextract.extract(url)
    tld = ext.suffix or ""
    subdomain = ext.subdomain or ""
    features["tld_in_path"] = 1 if tld in path else 0
    features["tld_in_subdomain"] = 1 if tld in subdomain else 0
    features["abnormal_subdomain"] = 1 if len(subdomain.split(".")) > 3 else 0
    features["nb_subdomains"] = len(subdomain.split(".")) if subdomain else 0
    features["prefix_suffix"] = 1 if "-" in hostname else 0
    features["random_domain"] = 1 if re.search(r"[a-z]{5,}\d+[a-z]*", hostname) else 0
    features["shortening_service"] = 1 if re.search(r"bit\.ly|goo\.gl|tinyurl|ow\.ly", url) else 0
    features["path_extension"] = 1 if re.search(r"\.[a-zA-Z]{2,5}$", path) else 0

    # --- Text metrics ---
    words = re.split(r"\W+", url)
    host_words = re.split(r"\W+", hostname)
    path_words = re.split(r"\W+", path)
    def safe_avg(lst): return sum(len(w) for w in lst if w) / max(1, len([w for w in lst if w]))
    features["length_words_raw"] = len(words)
    features["char_repeat"] = 1 if re.search(r"(.)\1{3,}", url) else 0
    features["shortest_words_raw"] = min([len(w) for w in words if w], default=0)
    features["shortest_word_host"] = min([len(w) for w in host_words if w], default=0)
    features["shortest_word_path"] = min([len(w) for w in path_words if w], default=0)
    features["longest_words_raw"] = max([len(w) for w in words if w], default=0)
    features["longest_word_host"] = max([len(w) for w in host_words if w], default=0)
    features["longest_word_path"] = max([len(w) for w in path_words if w], default=0)
    features["avg_words_raw"] = safe_avg(words)
    features["avg_word_host"] = safe_avg(host_words)
    features["avg_word_path"] = safe_avg(path_words)

    # --- Suspicious patterns ---
    features["phish_hints"] = 1 if re.search(r"login|secure|account|update|bank|free|verify", url.lower()) else 0
    features["domain_in_brand"] = 1 if ext.domain in url.lower() else 0
    features["brand_in_subdomain"] = 1 if any(b in subdomain for b in ["paypal", "amazon", "bank"]) else 0
    features["brand_in_path"] = 1 if any(b in path for b in ["paypal", "amazon", "bank"]) else 0
    features["suspecious_tld"] = 1 if tld in ["tk", "ml", "ga", "cf", "gq"] else 0
    features["statistical_report"] = 0

    # --- Content-based ---
    try:
        soup = BeautifulSoup(content, "html.parser") if content else BeautifulSoup(requests.get(url, timeout=5).text, "html.parser")
    except:
        soup = BeautifulSoup("", "html.parser")

    links = soup.find_all("a")
    nb_hyperlinks = len(links)
    features["nb_hyperlinks"] = nb_hyperlinks
    features["ratio_intHyperlinks"] = sum(1 for a in links if url in (a.get("href") or "")) / max(1, nb_hyperlinks)
    features["ratio_extHyperlinks"] = 1 - features["ratio_intHyperlinks"]
    features["ratio_nullHyperlinks"] = sum(1 for a in links if not a.get("href")) / max(1, nb_hyperlinks)
    features["nb_extCSS"] = len(soup.find_all("link", {"rel": "stylesheet"}))
    features["ratio_intRedirection"] = 0
    features["ratio_extRedirection"] = 0
    features["ratio_intErrors"] = 0
    features["ratio_extErrors"] = 0
    features["login_form"] = 1 if soup.find("input", {"type": "password"}) else 0
    features["external_favicon"] = 1 if "favicon" in str(soup) and "http" in str(soup) else 0
    features["links_in_tags"] = len(soup.find_all(["link", "script", "img"]))
    features["submit_email"] = 1 if re.search(r"mailto:", str(soup)) else 0
    features["ratio_intMedia"] = 0
    features["ratio_extMedia"] = 0
    features["sfh"] = 1 if soup.find("form", {"action": ""}) else 0
    features["iframe"] = 1 if soup.find("iframe") else 0
    features["popup_window"] = 1 if "popup" in content else 0
    features["safe_anchor"] = 1 if all("#" in (a.get("href") or "#") for a in links) else 0
    features["onmouseover"] = 1 if "onmouseover" in content else 0
    features["right_clic"] = 1 if "event.button==2" in content else 0
    features["empty_title"] = 1 if not soup.title else 0
    features["domain_in_title"] = 1 if hostname in (soup.title.string if soup.title else "") else 0
    features["domain_with_copyright"] = 1 if "©" in content or "copyright" in content.lower() else 0

    # --- WHOIS / Domain info ---
    try:
        domain_info = whois.whois(hostname)
        reg_date = domain_info.creation_date
        exp_date = domain_info.expiration_date
        if isinstance(reg_date, list): reg_date = reg_date[0]
        if isinstance(exp_date, list): exp_date = exp_date[0]
        features["domain_registration_length"] = (exp_date - reg_date).days if reg_date and exp_date else 0
        features["domain_age"] = (datetime.datetime.now() - reg_date).days if reg_date else 0
    except:
        features["domain_registration_length"] = 0
        features["domain_age"] = 0

    features["whois_registered_domain"] = 1 if hostname else 0
    features["web_traffic"] = 0
    features["dns_record"] = 1 if hostname else 0
    features["google_index"] = 0
    features["page_rank"] = 0
    features["status"] = 1

    # Return vector in correct order
    return [features[f] for f in FEATURE_ORDER]
