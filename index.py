import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
from chonkie import TokenChunker
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests

ENDPOINT_URL = (
    "https://script.google.com/macros/s/AKfycbxNEMCyCnnsbj_RjHSf9u564uIbLr1DtxFxQNIbU_KklYZkfZDm_9h60aDgYInLGevJCQ/exec"
)
EMBEDDING_MODEL = "gemini-embedding-2"
MAX_CHUNK_TOKENS = 8192
SUPPORTED_IMAGE_MIMES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
}

chunker = TokenChunker(chunk_size=MAX_CHUNK_TOKENS)


def load_env(workspace_dir: str):
  """Loads environment variables from .env if present."""
  env_path = os.path.join(workspace_dir, ".env")
  if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
  else:
    load_dotenv(override=True)


def get_repo_config(workspace_dir: str):
  """Derives repo OWNER and REPO from environment, returning (owner, repo, repo_url)."""
  load_env(workspace_dir)
  owner = os.environ.get("OWNER")
  repo = os.environ.get("REPO")
  if not owner or not repo:
    raise ValueError(
        "OWNER and REPO environment variables must be set in .env"
    )
  repo_url = f"https://github.com/{owner}/{repo}"
  return owner, repo, repo_url


def get_genai_client(workspace_dir: str):
  load_env(workspace_dir)
  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
  return genai.Client(api_key=api_key)


def get_github_headers() -> dict[str, str]:
  """Returns request headers for GitHub API, incorporating GITHUB_TOKEN if available."""
  headers = {
      "Accept": "application/vnd.github.raw+json",
      "User-Agent": "index/1.0",
  }
  token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
  if token:
    headers["Authorization"] = f"Bearer {token}"
  return headers


def make_github_request(
    url: str, headers: dict[str, str] = None
) -> requests.Response:
  """Makes a GET request to GitHub API with automatic rate-limit wait & retry."""
  if headers is None:
    headers = get_github_headers()

  while True:
    response = requests.get(url, headers=headers, allow_redirects=True, timeout=15)
    if response.status_code in (403, 429):
      remaining = response.headers.get("X-RateLimit-Remaining")
      if remaining == "0" or "rate limit" in response.text.lower():
        reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))
        limit = response.headers.get("X-RateLimit-Limit", "unknown")
        reset_str = (
            datetime.datetime.fromtimestamp(
                reset_timestamp, tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S UTC")
            if reset_timestamp
            else "unknown"
        )
        wait_secs = (
            max(1, reset_timestamp - int(time.time()) + 2)
            if reset_timestamp
            else 60
        )
        print(
            f"\n[RATE LIMIT] GitHub API rate limit reached (Limit: {limit})."
            f" Resetting at {reset_str}.\nSleeping for {wait_secs} seconds..."
        )
        time.sleep(wait_secs)
        print("[RATE LIMIT] Resuming request...")
        continue
    if response.status_code == 422:
      return response
    response.raise_for_status()
    return response


def prepare_query_and_document(content: str) -> str:
  """Prepares text with task prefix required by gemini-embedding-2."""
  return f"task: classification | query: {content}"


def get_chunk_suffix(chunk_idx: int) -> str:
  """Returns 'a', 'b', ..., 'z', 'aa', 'ab', etc. for chunk_idx (0-indexed)."""
  result = []
  while True:
    result.append(chr(ord("a") + (chunk_idx % 26)))
    chunk_idx = chunk_idx // 26 - 1
    if chunk_idx < 0:
      break
  return "".join(reversed(result))


def extract_image_urls(text: str) -> list[str]:
  """Extracts image URLs from HTML <img> tags and markdown ![alt](url) syntax."""
  urls = []
  # HTML <img> tags: <img ... src="URL" ...>
  html_matches = re.findall(
      r'<img\s+[^>]*?src=["\']([^"\']+)["\']', text, re.IGNORECASE
  )
  urls.extend(html_matches)

  # Markdown ![alt](url) syntax
  md_matches = re.findall(
      r'!\[[^\]]*\]\(([^)\s]+)(?:\s+["\'][^"\']*["\'])?\)', text
  )
  urls.extend(md_matches)

  # Deduplicate while preserving order
  seen = set()
  return [u for u in urls if not (u in seen or seen.add(u))]


def get_file_extension(mime_type: str, url: str) -> str:
  """Determines file extension from MIME type or URL."""
  mime_type = mime_type.lower()
  if "png" in mime_type:
    return "png"
  if "jpeg" in mime_type or "jpg" in mime_type:
    return "jpg"
  if "gif" in mime_type:
    return "gif"
  if "webp" in mime_type:
    return "webp"
  if "svg" in mime_type:
    return "svg"

  url_lower = url.lower()
  for ext in ["png", "jpg", "jpeg", "gif", "webp", "svg"]:
    if url_lower.endswith(f".{ext}"):
      return "jpg" if ext == "jpeg" else ext

  return "png"


def download_image(url: str):
  """Downloads an image from URL, returning (bytes, mime_type, extension) or None."""
  try:
    url_lower = url.lower()
    # Skip vector SVG images and shield badges which are not supported raster images
    if ".svg" in url_lower or "shields.io" in url_lower or "badge" in url_lower:
      return None

    headers = get_github_headers()
    response = make_github_request(url, headers=headers)
    if response.status_code != 200:
      return None

    content_type = response.headers.get("Content-Type", "")
    if ";" in content_type:
      content_type = content_type.split(";")[0].strip()

    content_type_lower = content_type.lower()
    if content_type_lower in ("image/svg+xml", "image/svg"):
      return None

    if not content_type_lower.startswith("image/"):
      if url_lower.endswith(".jpg") or url_lower.endswith(".jpeg"):
        content_type = "image/jpeg"
      elif url_lower.endswith(".gif"):
        content_type = "image/gif"
      elif url_lower.endswith(".webp"):
        content_type = "image/webp"
      elif url_lower.endswith(".png"):
        content_type = "image/png"
      else:
        return None

    if content_type.lower() not in SUPPORTED_IMAGE_MIMES:
      return None

    ext = get_file_extension(content_type, url)
    return response.content, content_type, ext
  except Exception as e:
    print(f"Warning: Failed to download image from {url}: {e}")
    return None


def fetch_all_open_issues(owner: str, repo: str) -> list[dict]:
  """Fetches ALL open issues (excluding pull requests) for owner/repo using pagination."""
  headers = get_github_headers()
  issues = []
  page = 1
  per_page = 100

  print(f"Fetching open issues list for {owner}/{repo}...")
  while True:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=open&per_page={per_page}&page={page}"
    response = make_github_request(url, headers=headers)
    if response.status_code == 422:
      print("Reached GitHub API offset pagination limit (1,000 items max).")
      break
    items = response.json()

    if not items:
      break

    page_issues = [item for item in items if "pull_request" not in item]
    issues.extend(page_issues)
    print(
        f"Fetched page {page} ({len(page_issues)} issues found). Total issues"
        f" fetched so far: {len(issues)}"
    )

    if len(items) < per_page:
      break
    page += 1

  return issues


def fetch_issue_comments(owner: str, repo: str, issue_number: int) -> list[dict]:
  """Fetches all comments for a given issue using pagination."""
  headers = get_github_headers()
  comments = []
  page = 1
  per_page = 100
  while True:
    comments_api_url = (
        f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page={per_page}&page={page}"
    )
    comments_response = make_github_request(comments_api_url, headers=headers)
    if comments_response.status_code == 422:
      break
    items = comments_response.json()
    if not items:
      break
    comments.extend(items)
    if len(items) < per_page:
      break
    page += 1
  return comments


def fetch_existing_hashes(endpoint_url: str = ENDPOINT_URL) -> set[str]:
  """Fetches set of existing content hashes from the endpoint via GET request."""
  print(f"Fetching existing content hashes from endpoint...", flush=True)
  try:
    response = requests.get(endpoint_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    hashes = set(data.get("hashes", []))
    print(f"Retrieved {len(hashes)} existing content hash(es) from vector database.", flush=True)
    return hashes
  except Exception as e:
    print(f"Warning: Failed to fetch existing hashes ({e}). Proceeding with empty set.", flush=True)
    return set()


def send_index(payload: dict, endpoint_url: str = ENDPOINT_URL) -> requests.Response:
  """Sends a POST request with the given payload to the endpoint URL with validation and retries."""
  headers = {
      "Content-Type": "text/plain;charset=utf-8",
  }
  max_retries = 5
  for attempt in range(1, max_retries + 1):
    try:
      response = requests.post(
          endpoint_url,
          data=json.dumps(payload),
          headers=headers,
          timeout=30,
      )
      response.raise_for_status()
      try:
        res_json = response.json()
        if not res_json.get("ok"):
          raise RuntimeError(f"Apps Script returned error: {res_json.get('error')}")
      except json.JSONDecodeError:
        pass
      return response
    except Exception as e:
      print(f"  [Attempt {attempt}/{max_retries}] POST request failed: {e}", flush=True)
      if attempt == max_retries:
        raise e
      time.sleep(5 * attempt)


def fetch_and_index_issue(
    owner: str,
    repo: str,
    issue_data: dict,
    genai_client: genai.Client,
    existing_hashes: set[str],
    endpoint_url: str = ENDPOINT_URL,
):
  issue_id = str(issue_data["id"])
  issue_number = issue_data["number"]

  # Fetch issue comments (subsequent replies)
  comments_data = fetch_issue_comments(owner, repo, issue_number)

  # Prepare list of items to process: main issue body first, then each comment reply
  items_to_process = [
      {
          "id": issue_id,
          "body": issue_data.get("body") or "",
          "label": "Issue Body",
      }
  ] + [
      {
          "id": str(comment["id"]),
          "body": comment.get("body") or "",
          "label": f"Comment #{comment['id']}",
      }
      for comment in comments_data
  ]

  print(
      f"Processing Issue #{issue_number} (ID: {issue_id}) with"
      f" {len(items_to_process)} item(s) (body + comments)..."
  )

  for idx, item in enumerate(items_to_process, start=1):
    comment_id = item["id"]
    comment_body = item["body"]
    label = item["label"]

    chunks = chunker.chunk(comment_body)
    chunk_texts = [c.text for c in chunks] if chunks else [comment_body]
    img_counter = 0

    for chunk_idx, chunk_text in enumerate(chunk_texts):
      suffix = get_chunk_suffix(chunk_idx)
      file_prefix = f"{idx}_{suffix}"

      # Compute SHA-256 hash for THIS specific chunk's source text
      chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

      # Check if chunk hash already exists in database
      if chunk_hash in existing_hashes:
        print(
            f"  Skipping {label} chunk {file_prefix} (hash {chunk_hash[:8]}... already exists in sheet).",
            flush=True,
        )
        continue

      # Format content with task instruction
      raw_text = chunk_text if chunk_text.strip() else " "
      formatted_text = prepare_query_and_document(raw_text)

      contents = [formatted_text]

      # Parse and attach embedded images
      image_urls = extract_image_urls(chunk_text)
      if image_urls:
        for img_url in image_urls:
          image_data = download_image(img_url)
          if image_data:
            img_bytes, mime_type, ext = image_data
            img_counter += 1

            part = types.Part.from_bytes(
                data=img_bytes,
                mime_type=mime_type,
            )
            contents.append(part)
            print(f"Attached image part from {img_url}")

      # Generate Gemini embedding with text-only fallback if image embedding is rejected
      try:
        embed_res = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=contents,
        )
      except Exception as e:
        if len(contents) > 1:
          print(
              f"Warning: Multimodal embedding failed for {file_prefix} ({e})."
              " Retrying with text-only content..."
          )
          embed_res = genai_client.models.embed_content(
              model=EMBEDDING_MODEL,
              contents=[formatted_text],
          )
        else:
          raise e

      embedding_values = embed_res.embeddings[0].values
      print(
          f"  Generated embedding for {label} chunk {file_prefix} ({len(embedding_values)} dims, hash {chunk_hash[:8]}...)"
      )

      # Send 1 POST request per chunk with that specific chunk's source hash & embedding vector
      payload = {
          "issue": issue_id,
          "id": comment_id,
          "hash": chunk_hash,
          "embedding": embedding_values,
      }

      print(
          f"Sending POST for issue '{payload['issue']}', comment '{payload['id']}', chunk {file_prefix}, hash '{payload['hash'][:8]}...'...",
          flush=True,
      )
      response = send_index(payload, endpoint_url)
      print(f"Status Code: {response.status_code}", flush=True)
      print(f"Response Content: {response.text}", flush=True)

      # Mark hash as existing to prevent duplicate processing
      existing_hashes.add(chunk_hash)


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
  owner, repo, repo_url = get_repo_config(workspace_dir)
  genai_client = get_genai_client(workspace_dir)

  print(f"Derived repository URL from .env: {repo_url}")
  try:
    existing_hashes = fetch_existing_hashes(ENDPOINT_URL)
    open_issues = fetch_all_open_issues(owner, repo)[:20]
    print(f"Found open issues. Processing first {len(open_issues)} issues for development...")

    for i, issue_data in enumerate(open_issues, start=1):
      print(f"\n--- Issue {i}/{len(open_issues)} (ID: #{issue_data['number']}) ---")
      fetch_and_index_issue(owner, repo, issue_data, genai_client, existing_hashes)
  except Exception as e:
    print(f"\n[ERROR] {e}", file=sys.stderr, flush=True)
    sys.exit(1)


if __name__ == "__main__":
  main()
