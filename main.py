"""
CLI that reverse-engineers top search results to merge their heading structures into a single, comprehensive 'Uber-Outli

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike Tools2U/AI-Website-Audit-CLI which only diagnosed existing HTML, this tool acts as a pre-mortem strategist; instead of checking if your tags are right, it tells you exactly what H2s and H3s you
"""
#!/usr/bin/env python3
"""
Uber-Outline Generator: Reverse-Engineers Top Search Results for Content Strategy.

This tool scrapes the top organic search results for a given keyword from DuckDuckGo,
extracts semantic heading structures (H1, H2, H3) from the HTML, and leverages an
LLM (OpenAI or Ollama) to synthesize these disparate structures into a single,
comprehensive, and logical "Uber-Outline" for creating high-quality content.

Usage:
    # Set environment variables (recommended)
    export OPENAI_API_KEY="sk-..."
    export OLLAMA_BASE_URL="http://localhost:11434"
    
    # Run with OpenAI (default)
    python uber_outline.py "machine learning mlops pipelines" -o mlops_outline.md

    # Run with Ollama
    python uber_outline.py "rust memory safety" --llm ollama --model mistral -o rust_guide.md

Dependencies:
    - requests (pip install requests)
    - standard library only otherwise
"""

import argparse
import os
import re
import sys
import json
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote, urljoin, urlparse
import textwrap
import subprocess
from http.client import IncompleteRead

try:
    import requests
except ImportError:
    print("Error: 'requests' library is missing.", file=sys.stderr)
    print("Install it using: pip install requests", file=sys.stderr)
    sys.exit(1)


# --- Constants & Configuration ---

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0.0.0 Safari/537.36"
)

DDG_SEARCH_URL = "https://html.duckduckgo.com/html/"

# System prompt for the LLM to ensure high-quality merging
MERGE_SYSTEM_PROMPT = """You are an expert technical editor and content strategist. 
Your task is to analyze a list of heading structures (H1, H2, H3) scraped from top-ranking web pages for a specific keyword.
Your goal is to synthesize these into ONE single, logical, and comprehensive "Uber-Outline".

Requirements:
1. **Hierarchy:** Establish a strict hierarchy. A single H1 (the title), followed by H2s, and H3s nested under relevant H2s.
2. **Comprehensiveness:** Merge unique sub-topics. If Page A mentions 'Security' and Page B mentions 'Compliance', create an H2 'Security and Compliance' with children H3s for both if they are distinct.
3. **Redundancy:** Remove exact duplicates.
4. **Flow:** Ensure the narrative flow makes sense for a reader (e.g., Introduction -> Core Concepts -> Implementation -> Best Practices -> Conclusion).
5. **Format:** Output strictly in Markdown format. Start with # Title.
6. **No Hallucination:** Do not add significant topics not present in the input, but you are allowed to rephrase headers for clarity and consistency.
"""

# --- Helper Functions ---

def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Safely retrieves an environment variable."""
    return os.environ.get(key, default)

def clean_text(text: str) -> str:
    """Removes extra whitespace and newlines."""
    return " ".join(text.split())

def is_valid_url(url: str) -> bool:
    """Basic URL validation."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

# --- Scraping Logic ---

def fetch_ddg_results(keyword: str, max_results: int = 5) -> List[str]:
    """
    Fetches organic result URLs from DuckDuckGo HTML search.
    
    Args:
        keyword: The search query.
        max_results: Number of links to return.

    Returns:
        A list of unique, valid URLs.
    """
    print(f"[*] Searching DuckDuckGo for: '{keyword}'...")
    params = {"q": keyword}
    headers = {"User-Agent": USER_AGENT}
    
    urls = []
    try:
        response = requests.get(DDG_SEARCH_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        
        # DuckDuckGo HTML parsing is tricky; regex is fragile but safer than BS4 here given dependencies.
        # We look for the standard <a rel="nofollow" class="result__url" ...> structure or similar.
        # A more robust approach uses regex to find links inside result snippets.
        
        # Pattern to find result URLs in DDG HTML source
        # Usually looks like: <a rel="nofollow" href="https://example.com" class="result__a">
        link_pattern = re.compile(r'<a[^>]+href=["\'](https?://[^"\']+)["\'][^>]*class=["\']result__a["\']', re.IGNORECASE)
        
        matches = link_pattern.findall(response.text)
        
        for url in matches:
            if len(urls) >= max_results:
                break
            # DDG sometimes redirects, but html.duckduckgo.com usually provides direct links in this view
            # or laddered links. We'll do a basic filter.
            if is_valid_url(url) and "duckduckgo" not in url:
                if url not in urls:
                    urls.append(url)
                    
    except requests.RequestException as e:
        print(f"[!] Error fetching search results: {e}", file=sys.stderr)
        
    return urls

def extract_headers_from_html(html_content: str) -> List[str]:
    """
    Parses HTML content and extracts text from H1, H2, and H3 tags.
    
    Args:
        html_content: Raw HTML string.

    Returns:
        List of header strings.
    """
    # Using stdlib html.parser to avoid BeautifulSoup dependency
    from html.parser import HTMLParser
    
    class HeaderParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.headers = []
            self.current_tag = None
            
        def handle_starttag(self, tag, attrs):
            if tag in ['h1', 'h2', 'h3']:
                self.current_tag = tag
        
        def handle_endtag(self, tag):
            if tag in ['h1', 'h2', 'h3']:
                self.current_tag = None
                
        def handle_data(self, data):
            if self.current_tag:
                clean = clean_text(data)
                if clean and len(clean) > 2: # Filter out short junk
                    # Prepend the tag level for LLM context
                    self.headers.append(f"{self.current_tag.upper()}: {clean}")
    
    parser = HeaderParser()
    try:
        parser.feed(html_content)
    except Exception as e:
        print(f"[!] Warning: HTML parsing error on page: {e}", file=sys.stderr)
        
    return parser.headers

def scrape_urls(urls: List[str]) -> Dict[str, List[str]]:
    """
    Iterates through URLs and extracts headers.
    
    Args:
        urls: List of target URLs.
        
    Returns:
        Dictionary mapping URL -> List of Extracted Headers.
    """
    results = {}
    headers = {"User-Agent": USER_AGENT}
    
    for i, url in enumerate(urls, 1):
        print(f"[*] Scraping {i}/{len(urls)}: {url}")
        try:
            # Timeout after 10 seconds
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            headers_found = extract_headers_from_html(r.text)
            if headers_found:
                results[url] = headers_found
                print(f"    - Found {len(headers_found)} headers.")
            else:
                print(f"    - No headers found (might be JS rendered or blocked).")
        except IncompleteRead:
            print(f"    - Connection incomplete.", file=sys.stderr)
        except requests.RequestException as e:
            print(f"    - Error fetching URL: {e}", file=sys.stderr)
        except Exception as e:
            print(f"    - Unexpected error: {e}", file=sys.stderr)
            
    return results

def format_headers_for_llm(scraped_data: Dict[str, List[str]]) -> str:
    """Formats the scraped data into a string prompt for the LLM."""
    output_tokens = []
    output_tokens.append("Here are the heading structures from the top search results:\n")
    
    for url, headers in scraped_data.items():
        output_tokens.append(f"--- SOURCE: {url} ---")
        for h in headers:
            output_tokens.append(h)
        output_tokens.append("") # Spacer
        
    return "\n".join(output_tokens)

# --- LLM Integration ---

def call_openai(api_key: str, content: str, model: str = "gpt-3.5-turbo") -> str:
    """Calls the OpenAI Chat Completions API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "temperature": 0.5
    }
    
    try:
        print(f"[*] Querying OpenAI model '{model}'...")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        response_json = response.json()
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            raise ValueError("Invalid response structure from OpenAI")
            
    except requests.RequestException as e:
        raise RuntimeError(f"OpenAI API request failed: {e}")

def call_ollama(base_url: str, content: str, model: str = "llama3") -> str:
    """
    Calls the local Ollama instance via its API.
    Ollama usually runs on http://localhost:11434/api/generate or /api/chat
    """
    # Ensure base_url doesn't have trailing slash
    base_url = base_url.rstrip('/')
    endpoint = f"{base_url}/api/chat"
    
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ],
        "options": {"temperature": 0.5}
    }
    
    try:
        print(f"[*] Querying Ollama model '{model}' at {base_url}...")
        response = requests.post(endpoint, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        response_json = response.json()
        
        if "message" in response_json:
            return response_json["message"]["content"]
        else:
            raise ValueError("Invalid response structure from Ollama")
            
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Failed to connect to Ollama at {base_url}. "
            "Ensure Ollama is running and the URL is correct."
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama API request failed: {e}")

def generate_llm_outline(scraped_content: str, provider: str, model: str, api_key: Optional[str] = None) -> str:
    """Dispatches the request to the appropriate provider."""
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required for provider 'openai'.")
        return call_openai(api_key, scraped_content, model)
    elif provider == "ollama":
        # Ollama doesn't strictly need an API key, but we use the env var for the base URL usually
        url = get_env_var("OLLAMA_BASE_URL", "http://localhost:11434")
        return call_ollama(url, scraped_content, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(
        description="Uber-Outline: Reverse-engineers search results into a perfect content structure.",
        epilog="Example: python uber_outline.py 'content marketing 2024' -o outline.md"
    )
    
    parser.add_argument("keyword", help="Target keyword to search for.")
    parser.add_argument("-o", "--output", default="uber_outline.md", help="Output Markdown filename.")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai", 
                        help="LLM Provider (default: openai).")
    parser.add_argument("--model", help="Model name (e.g., gpt-4o, mistral). Defaults per provider.")
    parser.add_argument("--api-key", help="API Key. Defaults to OPENAI_API_KEY env var.")
    
    args = parser.parse_args()
    
    # 1. Validate Configuration
    api_key = args.api_key or get_env_var("OPENAI_API_KEY")
    if args.provider == "openai" and not api_key:
        print("[!] No OpenAI API key found. Set OPENAI_API_KEY env var or use --api-key.", file=sys.stderr)
        sys.exit(1)
        
    default_models = {
        "openai": "gpt-3.5-turbo",
        "ollama": "llama3"
    }
    model = args.model or default_models[args.provider]

    # 2. Search and Scrape
    urls = fetch_ddg_results(args.keyword, max_results=5)
    if not urls:
        print("[!] No search results found. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    scraped_data = scrape_urls(urls)
    if not scraped_data:
        print("[!] No headers could be extracted from any search results. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    # 3. Prepare LLM Input
    print("[*] Preparing data for LLM synthesis...")
    prompt_content = format_headers_for_llm(scraped_data)
    
    # 4. Generate Outline
    try:
        final_outline = generate_llm_outline(prompt_content, args.provider, model, api_key)
    except Exception as e:
        print(f"[!] Critical error during LLM generation: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Output to File
    try:
        # Wrap in a nice header in the file itself
        file_content = textwrap.dedent(f"""\
        # Generated Uber-Outline
            
        **Keyword:** {args.keyword}  
        **Sources Analyzed:** {len(scraped_data)}  
        **Model:** {args.provider}/{model}  
        
        ---
        
        {final_outline}
        """)
        
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(file_content)
            
        print(f"\n[+] Success! Uber-Outline saved to: {args.output}")
        
    except IOError as e:
        print(f"[!] Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()