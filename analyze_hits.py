import json
from collections import defaultdict

def analyze(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    providers = defaultdict(list)
    for entry in data:
        providers[entry['provider']].append(entry['gold_hit']['hit'])

    for p, hits in providers.items():
        print(f"{p}: {sum(hits)}/ {len(hits)}")

if __name__ == "__main__":
    import sys
    analyze(sys.argv[1])
