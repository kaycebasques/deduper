import json
import os
import re
import sys
import urllib.parse
from dotenv import load_dotenv
from google import genai
import requests

DEFAULT_ISSUE_URL = "https://github.com/sphinx-doc/sphinx/issues/14541"
EMBEDDING_MODEL = "gemini-embedding-001"


def parse_github_issue_url(url: str):
  """Parses a GitHub issue URL into (owner, repo, issue_number)."""
  parsed = urllib.parse.urlparse(url)
  match = re.match(r"^/([^/]+)/([^/]+)/issues/(\d+)", parsed.path)
  if not match:
    raise ValueError(f"Invalid GitHub issue URL: {url}")
  return match.group(1), match.group(2), match.group(3)


def get_genai_client(workspace_dir: str):
  env_path = os.path.join(workspace_dir, ".env")
  if os.path.exists(env_path):
    load_dotenv(env_path)
  else:
    load_dotenv()
  api_key = os.environ.get("GEMINI_API_KEY")
  if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
  return genai.Client(api_key=api_key)


def fetch_and_save_issue(
    url: str, workspace_dir: str, genai_client: genai.Client
):
  owner, repo, issue_number = parse_github_issue_url(url)

  headers = {
      "Accept": "application/vnd.github.raw+json",
      "User-Agent": "deduper/1.0",
  }

  # Fetch issue details (opening post / issue body)
  issue_api_url = (
      f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
  )
  response = requests.get(issue_api_url, headers=headers)
  response.raise_for_status()
  issue_data = response.json()

  # Create issues/{issue_number} directory
  issue_dir = os.path.join(workspace_dir, "issues", str(issue_number))
  os.makedirs(issue_dir, exist_ok=True)

  # Fetch issue comments (subsequent replies)
  comments_api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=100"
  comments_response = requests.get(comments_api_url, headers=headers)
  comments_response.raise_for_status()
  comments_data = comments_response.json()

  # List of all comments (1 is issue body, 2..N are replies)
  all_comments = [issue_data.get("body") or ""] + [
      c.get("body") or "" for c in comments_data
  ]

  for idx, comment_body in enumerate(all_comments, start=1):
    # Save markdown file
    md_path = os.path.join(issue_dir, f"{idx}.md")
    with open(md_path, "w", encoding="utf-8") as f:
      f.write(comment_body)
    print(f"Saved comment {idx} markdown to {md_path}")

    # Generate Gemini embedding
    print(f"Generating embedding for comment {idx}...")
    embed_res = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=comment_body if comment_body.strip() else " ",
    )
    embedding_values = embed_res.embeddings[0].values

    # Save JSON embedding file
    json_path = os.path.join(issue_dir, f"{idx}.json")
    with open(json_path, "w", encoding="utf-8") as f:
      json.dump(embedding_values, f)
    print(
        f"Saved comment {idx} embedding ({len(embedding_values)} dims) to"
        f" {json_path}"
    )


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
  genai_client = get_genai_client(workspace_dir)
  issue_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ISSUE_URL
  fetch_and_save_issue(issue_url, workspace_dir, genai_client)


if __name__ == "__main__":
  main()
