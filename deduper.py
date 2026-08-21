import os
import re
import sys
import urllib.parse
import requests

DEFAULT_ISSUE_URL = "https://github.com/sphinx-doc/sphinx/issues/14541"


def parse_github_issue_url(url: str):
  """Parses a GitHub issue URL into (owner, repo, issue_number)."""
  parsed = urllib.parse.urlparse(url)
  match = re.match(r"^/([^/]+)/([^/]+)/issues/(\d+)", parsed.path)
  if not match:
    raise ValueError(f"Invalid GitHub issue URL: {url}")
  return match.group(1), match.group(2), match.group(3)


def fetch_and_save_issue(url: str, workspace_dir: str):
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

  # 1.md is the main issue description (first comment)
  first_comment_body = issue_data.get("body") or ""
  first_comment_path = os.path.join(issue_dir, "1.md")
  with open(first_comment_path, "w", encoding="utf-8") as f:
    f.write(first_comment_body)
  print(f"Saved first comment to {first_comment_path}")

  # Fetch issue comments (subsequent replies)
  comments_api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments?per_page=100"
  comments_response = requests.get(comments_api_url, headers=headers)
  comments_response.raise_for_status()
  comments_data = comments_response.json()

  for idx, comment in enumerate(comments_data, start=2):
    comment_body = comment.get("body") or ""
    comment_path = os.path.join(issue_dir, f"{idx}.md")
    with open(comment_path, "w", encoding="utf-8") as f:
      f.write(comment_body)
    print(f"Saved comment {idx} to {comment_path}")


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
  issue_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ISSUE_URL
  fetch_and_save_issue(issue_url, workspace_dir)


if __name__ == "__main__":
  main()
