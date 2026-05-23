import json
from collections import defaultdict

def analyze(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    queries = defaultdict(list)
    for entry in data:
        queries[entry['query_id']].append(entry)

    diffs = []
    for q_id, entries in queries.items():
        liner = next((e for e in entries if e['provider'] == 'liner'), None)
        perp = next((e for e in entries if e['provider'] == 'perplexity'), None)
        
        if liner and perp:
            # We want cases where Perplexity hit gold but Liner didn't, or vice versa
            lp_hit = liner['gold_hit']['hit']
            pp_hit = perp['gold_hit']['hit']
            
            if pp_hit and not lp_hit:
                diffs.append({
                    'query_id': q_id,
                    'query': liner['query'],
                    'liner_results': liner['results'],
                    'perp_results': perp['results']
                })

    return diffs

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    diffs = analyze(path)
    print(f"Found {len(diffs)} cases where Perplexity hit gold but Liner didn't.")
    for d in diffs[:3]: # Show first 3 examples
        print(f"\nQuery ID: {d['query_id']}")
        print(f"Query: {d['query']}")
        print("Liner Top-1 URL:", d['liner_results'][0]['url'] if d['liner_results'] else "None")
        print("Perplexity Top-1 URL:", d['perp_results'][0]['url'] if d['perp_results'] else "None")

