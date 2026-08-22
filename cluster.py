import collections
import json
import operator
import os
import sys
from dotenv import load_dotenv


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


def dot_product(v1: list[float], v2: list[float]) -> float:
  """Calculates dot product between two embedding vectors."""
  return sum(map(operator.mul, v1, v2))



class UnionFind:
  """Disjoint Set Union (DSU) structure to group issues into clusters."""

  def __init__(self, elements):
    self.parent = {x: x for x in elements}
    self.rank = {x: 0 for x in elements}

  def find(self, x):
    if self.parent[x] != x:
      self.parent[x] = self.find(self.parent[x])
    return self.parent[x]

  def union(self, x, y):
    root_x = self.find(x)
    root_y = self.find(y)
    if root_x != root_y:
      if self.rank[root_x] < self.rank[root_y]:
        root_x, root_y = root_y, root_x
      self.parent[root_y] = root_x
      if self.rank[root_x] == self.rank[root_y]:
        self.rank[root_x] += 1


def load_issue_embeddings(workspace_dir: str) -> dict[str, list[float]]:
  """Loads the embedding of the first comment (1_a.json) for each issue in issues/ directory."""
  issues_dir = os.path.join(workspace_dir, "issues")
  if not os.path.exists(issues_dir):
    raise FileNotFoundError(
        f"Issues directory not found at: {issues_dir}"
    )

  issue_ids = sorted(
      [
          d
          for d in os.listdir(issues_dir)
          if os.path.isdir(os.path.join(issues_dir, d))
      ],
      key=lambda x: int(x) if x.isdigit() else x,
  )

  embeddings = {}
  for issue_id in issue_ids:
    file_path = os.path.join(issues_dir, issue_id, "1_a.json")
    if os.path.exists(file_path):
      try:
        with open(file_path, "r", encoding="utf-8") as f:
          embeddings[issue_id] = json.load(f)
      except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")

  return embeddings


def build_clusters(
    embeddings: dict[str, list[float]], threshold: float = 0.9
) -> list[list[str]]:
  """Clusters issues based on embedding dot product similarity above the threshold."""
  issue_ids = list(embeddings.keys())
  vecs = [embeddings[id_] for id_ in issue_ids]
  n = len(issue_ids)
  uf = UnionFind(issue_ids)

  total_pairs = n * (n - 1) // 2
  print(f"Comparing {total_pairs} pairs...", flush=True)

  for i in range(n):
    v_i = vecs[i]
    id_i = issue_ids[i]
    for j in range(i + 1, n):
      if dot_product(v_i, vecs[j]) > threshold:
        uf.union(id_i, issue_ids[j])

  grouped = collections.defaultdict(list)
  for issue_id in issue_ids:
    root = uf.find(issue_id)
    grouped[root].append(issue_id)

  clusters = [
      sorted(members, key=lambda x: int(x) if x.isdigit() else x)
      for members in grouped.values()
      if len(members) > 1
  ]

  clusters.sort(key=lambda c: int(c[0]) if c[0].isdigit() else c[0])
  return clusters


def main():
  workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
  try:
    owner, repo, repo_url = get_repo_config(workspace_dir)
  except Exception as e:
    print(f"[ERROR] Configuration error: {e}", flush=True)
    sys.exit(1)

  print(
      f"Loading issue embeddings from: {os.path.join(workspace_dir, 'issues')}",
      flush=True,
  )
  embeddings = load_issue_embeddings(workspace_dir)
  print(f"Loaded embeddings for {len(embeddings)} issues.", flush=True)

  print("Clustering issues based on dot product similarity > 0.9...", flush=True)
  clusters = build_clusters(embeddings, threshold=0.9)

  print(f"\nFound {len(clusters)} issue cluster(s) (similarity > 0.9):\n", flush=True)
  for idx, cluster in enumerate(clusters, start=1):
    print(f"--- Cluster {idx} ({len(cluster)} issues) ---", flush=True)
    for issue_id in cluster:
      url = f"{repo_url}/issues/{issue_id}"
      print(f"  - Issue #{issue_id}: {url}", flush=True)
    print(flush=True)



if __name__ == "__main__":
  main()
