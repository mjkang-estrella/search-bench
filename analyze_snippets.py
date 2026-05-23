import json

def analyze(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if entry['provider'] == 'liner':
                # Just print a few results to see the snippets
                print(f"Query: {entry['query']}")
                print(f"Gold hit: {entry['gold_hit']['hit']}, Rank: {entry['gold_hit'].get('rank', 'N/A')}")
                for res in entry['results'][:3]:
                    print(f"Rank {res['rank']}: {res['title']} | Snippet: {res['snippet']}")
                print("-" * 40)
                break

if __name__ == "__main__":
    import sys
    analyze(sys.argv[1])
