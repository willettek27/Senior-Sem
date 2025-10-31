# =========================================
# extract_features.py
# Extract numeric URL-based features for phishing detection
# =========================================

import re
import tldextract
import urllib.parse as urlparse

# -----------------------------------------
# Feature order (from least to most important - 45 features)
# -----------------------------------------
FEATURES_ORDER = [
    "nb_at", "nb_com", "tld_in_path", "random_domain",
    "http_in_path", "whois_registered_domain", "tld_in_subdomain", "ip",
    "ratio_intMedia", "ratio_extMedia", "ratio_extErrors", "domain_with_copyright",
    "shortest_words_raw", "nb_redirection", "domain_in_brand", "nb_subdomains",
    "nb_extCSS", "nb_eq", "nb_qm", "avg_word_host", "nb_hyphens",
    "avg_words_raw", "ratio_digits_host", "shortest_word_path", "nb_slash",
    "domain_in_title", "domain_registration_length", "shortest_word_host",
    "links_in_tags", "avg_word_path", "length_hostname", "longest_words_raw",
    "ratio_digits_url", "length_url", "ratio_extRedirection", "safe_anchor",
    "ratio_intHyperlinks", "longest_word_path", "phish_hints", "ratio_extHyperlinks",
    "domain_age", "nb_www", "web_traffic", "page_rank", "google_index"
]

# -----------------------------------------
# Feature extraction
# -----------------------------------------
def extract_features(sample):
    url = sample.get("url", "")
    features = {}

    parsed = urlparse.urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # --- URL BASIC INFO ---
    features["length_url"] = len(url)
    features["length_hostname"] = len(hostname)
    features["ip"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0
    features["ratio_digits_url"] = len(re.findall(r"\d", url)) / max(1, len(url))
    features["ratio_digits_host"] = (
        sum(c.isdigit() for c in hostname) / max(1, len(hostname)) if hostname else 0
    )

    # --- Character counts ---
    features["nb_slash"] = url.count("/")
    features["nb_hyphens"] = url.count("-")
    features["nb_qm"] = url.count("?")
    features["nb_eq"] = url.count("=")
    features["nb_com"] = url.count(".com")
    features["nb_at"] = url.count("@")
    features["nb_www"] = url.count("www")

    # --- Domain and TLD ---
    ext = tldextract.extract(url)
    tld = ext.suffix or ""
    subdomain = ext.subdomain or ""

    features["tld_in_path"] = 1 if tld in path else 0
    features["tld_in_subdomain"] = 1 if tld in subdomain else 0
    features["nb_subdomains"] = len(subdomain.split(".")) if subdomain else 0
    features["random_domain"] = 1 if re.search(r"[a-z]{5,}\d+[a-z]*", hostname) else 0
    features["domain_in_brand"] = 1 if ext.domain in url.lower() else 0
    features["http_in_path"] = 1 if "http" in path else 0

    # --- Text-based metrics ---
    words = re.split(r"\W+", url)
    host_words = re.split(r"\W+", hostname)
    path_words = re.split(r"\W+", path)
    safe_avg = lambda lst: sum(len(w) for w in lst if w) / max(1, len([w for w in lst if w]))

    features["avg_words_raw"] = safe_avg(words)
    features["avg_word_host"] = safe_avg(host_words)
    features["avg_word_path"] = safe_avg(path_words)
    features["shortest_words_raw"] = min([len(w) for w in words if w], default=0)
    features["shortest_word_host"] = min([len(w) for w in host_words if w], default=0)
    features["shortest_word_path"] = min([len(w) for w in path_words if w], default=0)
    features["longest_words_raw"] = max([len(w) for w in words if w], default=0)
    features["longest_word_path"] = max([len(w) for w in path_words if w], default=0)

    # --- Heuristic / keyword-based ---
    features["phish_hints"] = 1 if re.search(r"login|secure|account|update|bank|free|verify", url.lower()) else 0

    # --- Placeholder / advanced (require HTML or WHOIS) ---
    placeholder_features = {
        "ratio_extHyperlinks": 0,
        "ratio_intHyperlinks": 0,
        "safe_anchor": 0,
        "ratio_extRedirection": 0,
        "links_in_tags": 0,
        "domain_in_title": 0,
        "domain_with_copyright": 0,
        "ratio_extErrors": 0,
        "ratio_extMedia": 0,
        "ratio_intMedia": 0,
        "domain_registration_length": 0,
        "domain_age": 0,
        "web_traffic": 0,
        "whois_registered_domain": 1 if hostname else 0,
        "google_index": 0,
        "page_rank": 0,
        "nb_extCSS": 0,
        "nb_redirection": 0
    }
    features.update(placeholder_features)

    # --- Ensure all features exist ---
    for f in FEATURES_ORDER:
        if f not in features:
            features[f] = 0

    return [features[f] for f in FEATURES_ORDER]
