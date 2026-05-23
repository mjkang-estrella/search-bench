import json

def analyze(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    queries = {}
    for entry in data:
        if entry['provider'] in ['liner', 'perplexity']:
            qid = entry['query_id']
            if qid not in queries:
                queries[qid] = {}
            queries[qid][entry['provider']] = entry

    # Find a query where they both hit gold but Liner isPartial and Perplexity is Answerable (approx)
    # Since we don't have the judge's detailed per-query score here, let's just pick one.
    q_id = list(queries.keys())[0]
    q = queries[q_id]
    print(f"Query ID: {q_id}")
    print(f"Query: {q['liner']['query']}")
    print("\n--- LINER SNIPPETS ---")
    for res in q['liner']['results'][:3]:
        print(f"Rank {res['rank']}: {res['title']} | Snippet: {res['snippet']}")
    print("\n--- PERPLEXITY SNIPPETS ---")
    for res in q['perplexity']['results'][:3]:
        print(f"Rank {res['rank']}: {res['title']} | Snippet: {res['snippet']}")

if __name__ == "__main__":
    import sys
    analyze(sys.argv[1])
