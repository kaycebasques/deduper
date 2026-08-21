import json
import os
import re
import sys
import urllib.parse
from dotenv import load_dotenv
from google import genai
import requests

EMBEDDING_MODEL = "gemini-embedding-001"
MAX_ISSUES = 5


def load_env(workspace_dir: str):
  """Loads environment variables from .env if present."""
  env_path = os.path.join(workspace_dir, ".env")
  if os.path.exists(env_path):
    load_dotenv(env_path)
  else:
    load_dotenv()


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


def fetch_open_issues(owner: str, repo: str, max_issues: int = MAX_ISSUES):
  """Fetches open issues (excluding pull requests) for owner/repo."""
  headers = {
      "Accept": "application/vnd.github.raw+json",
      "User-Agent": "deduper/1.0",
  }
  url = f"https://api.github.com/repos/{owner}/{repo}/issues?state=open&per_page=30"
  response = requests.get(url, headers=headers)
  response.raise_for_status()
  items = response.json()

  issues = [item for item in items if "pull_request" not in item]
  return issues[:max_issues]


def fetch_and_save_issue(
    owner: str,
    repo: str,
    issue_data: dict,
    workspace_dir: str,
    genai_client: genai.Client,
):
  issue_number = issue_data["number"]
  headers = {
      "Accept": "application/vnd.github.raw+json",
      "User-Agent": "deduper/1.0",
  }

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

  print(f"\nProcessing Issue #{issue_number} ({len(all_comments)} comments)...")
  for idx, comment_body in enumerate(all_comments, start=1):
    # Save markdown file
    md_path = os.path.join(issue_dir, f"{idx}.md")
    with open(md_path, "w", encoding="utf-8") as f:
      f.write(comment_body)
    print(f"Saved issue #{issue_number} comment {idx} markdown to {md_path}")

    # Generate Gemini embedding
    print(f"Generating embedding for issue #{issue_number} comment {idx}...")
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
        f"Saved issue #{issue_number} comment {idx} embedding"
        f" ({len(embedding_values)} dims) to {json_path}"
    )


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
  owner, repo, repo_url = get_repo_config(workspace_dir)
  genai_client = get_genai_client(workspace_dir)

  print(f"Derived repository URL from .env: {repo_url}")
  print(f"Fetching open issues for {owner}/{repo} (max {MAX_ISSUES})...")

  open_issues = fetch_open_issues(owner, repo, max_issues=MAX_ISSUES)
  print(f"Found {len(open_issues)} open issues to process.")

  for issue_data in open_issues:
    fetch_and_save_issue(owner, repo, issue_data, workspace_dir, genai_client)


if __name__ == "__main__":
  main()
