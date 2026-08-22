import json
import os
import sys
from dotenv import load_dotenv
import requests

ENDPOINT_URL = (
    "https://script.google.com/macros/s/AKfycbzWd-b3W_iFhbTcr-Y9jwS2r8f-DISudNOYMEm6i-qqZc3Yf2x3fmYPRA-wLWIVo-R0BA/exec"
)


def load_env(workspace_dir: str):
  """Loads environment variables from .env if present."""
  env_path = os.path.join(workspace_dir, ".env")
  if os.path.exists(env_path):
    load_dotenv(env_path)
  else:
    load_dotenv()


def send_index(payload: dict, endpoint_url: str = ENDPOINT_URL) -> requests.Response:
  """Sends a POST request with the given payload to the endpoint URL."""
  headers = {
      "Content-Type": "text/plain;charset=utf-8",
  }
  response = requests.post(
      endpoint_url,
      data=json.dumps(payload),
      headers=headers,
  )
  return response


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
  load_env(workspace_dir)

  payload = {
      "id": "foo",
      "embeddings": [
          [1, 2, 3],
          [4, 5, 6],
      ],
  }

  print(f"Sending POST request to {ENDPOINT_URL}...", flush=True)
  try:
    response = send_index(payload)
    print(f"Status Code: {response.status_code}", flush=True)
    print(f"Response Content: {response.text}", flush=True)
  except Exception as e:
    print(f"[ERROR] Request failed: {e}", file=sys.stderr, flush=True)
    sys.exit(1)


if __name__ == "__main__":
  main()
