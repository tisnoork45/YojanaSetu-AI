
import json
import retrieval_pipeline as rp

EVAL_DATASET_PATH = "data/eval_dataset.json"
THRESHOLD_SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0.20, 0.25, ..., 0.60
NO_THRESHOLD = -1.01  # cosine similarity is always >= -1.0, so this never filters anything out
REPRESENTATIVE_QUERIES = [
    "I need money for my crops",
    "I'm a 65 year old farmer looking for a pension",
    "What's the weather like today?",
]

def raw_ranked_results(query, embedding_model, index, df):
    return rp.retrieve_candidates(query, embedding_model, index, df,
                                   top_k=index.ntotal, threshold=NO_THRESHOLD)


def evaluate_raw_recall_at_10(retrieve_cases, embedding_model, index, df):
    hits = 0
    per_case = []
    for case in retrieve_cases:
        ranked = raw_ranked_results(case["query"], embedding_model, index, df)
        names = [c["scheme_name"] for c in ranked]
        if case["expected_scheme"] in names:
            rank = names.index(case["expected_scheme"]) + 1
            score = ranked[rank - 1]["_similarity"]
            in_top10 = rank <= 10
            hits += 1 if in_top10 else 0
            per_case.append({"id": case["id"], "expected_scheme": case["expected_scheme"],
                              "rank": rank, "score": score, "in_top10": in_top10})
        else:
            per_case.append({"id": case["id"], "expected_scheme": case["expected_scheme"],
                              "rank": None, "score": None, "in_top10": False})
    recall = hits / len(retrieve_cases) if retrieve_cases else 0.0
    return recall, per_case

def threshold_sweep_report(retrieve_cases, no_match_cases, embedding_model, index, df):
    print(f"{'threshold':>10} | {'recall@10':>10} | {'no-match FP rate':>18}")
    print("-" * 46)
    for threshold in THRESHOLD_SWEEP:
        hits = 0
        for case in retrieve_cases:
            candidates = rp.retrieve_candidates(case["query"], embedding_model, index, df, threshold=threshold)
            names = {c["scheme_name"] for c in candidates}
            if case["expected_scheme"] in names:
                hits += 1
        recall = hits / len(retrieve_cases) if retrieve_cases else 0.0

        false_positives = 0
        for case in no_match_cases:
            candidates = rp.retrieve_candidates(case["query"], embedding_model, index, df, threshold=threshold)
            if candidates:
                false_positives += 1
        fp_rate = false_positives / len(no_match_cases) if no_match_cases else 0.0

        print(f"{threshold:>10.2f} | {recall:>10.2%} | {fp_rate:>18.2%}")

def print_representative_queries(embedding_model, index, df):
    for query in REPRESENTATIVE_QUERIES:
        ranked = raw_ranked_results(query, embedding_model, index, df)[:10]
        print(f"\nQUERY: {query}")
        for rank, c in enumerate(ranked, start=1):
            print(f"  {rank:>2}. {c['_similarity']:.4f}  {c['scheme_name']}")

def per_case_threshold_table(per_case_raw):
    col_labels = "  ".join(f"{t:.2f}" for t in THRESHOLD_SWEEP)
    print(f"\n{'id':<10}{'rank':>5}{'score':>8}   {col_labels}")
    for row in per_case_raw:
        if row["score"] is None:
            print(f"{row['id']:<10}{'--':>5}{'--':>8}   (expected scheme not found in the index at all -- check the name/spelling)")
            continue
        if not row["in_top10"]:
            flags = "  ".join(" -- " for _ in THRESHOLD_SWEEP)
        else:
            flags = "  ".join(" Y  " if row["score"] >= t else " .  " for t in THRESHOLD_SWEEP)
        print(f"{row['id']:<10}{row['rank']:>5}{row['score']:>8.4f}   {flags}")
    print("\nY = would be retrieved at that threshold (real top_k=10 pipeline)")
    print(". = in top-10, but score falls below that threshold")
    print("-- = NOT in the raw top-10 at all -- no threshold can fix this case; see section 1")

def main():
    with open(EVAL_DATASET_PATH) as f:
        cases = json.load(f)  

    retrieve_cases = [c for c in cases if c["expected_behavior"] == "RETRIEVE" and c["expected_scheme"]]
    no_match_cases = [c for c in cases if c["category"] == "no_match"]

    try:
        index, df, embedding_model = rp.load_resources()
    except Exception as e:
        raise SystemExit(
            f"Could not load the FAISS store ({e}).\n"
            "Build it first with:\n"
            "    python data_cleaning.py\n"
            "    python enrich_eligibility.py\n"
            "    python build_index.py"
        )

    print("=" * 72)
    print("1. RAW (UNFILTERED) RECALL@10 -- ceiling before any threshold")
    print("=" * 72)
    raw_recall, per_case_raw = evaluate_raw_recall_at_10(retrieve_cases, embedding_model, index, df)
    print(f"Raw Recall@10: {raw_recall:.2%} ({len(retrieve_cases)} cases)")
    print("No threshold value can score higher than this on its own -- it's the")
    print("retrieval ceiling before any similarity cutoff is applied.")

    print("\n" + "=" * 72)
    print("2. THRESHOLD SWEEP: 0.20 to 0.60 in steps of 0.05")
    print("=" * 72)
    threshold_sweep_report(retrieve_cases, no_match_cases, embedding_model, index, df)

    print("\n" + "=" * 72)
    print("3. RAW TOP-10 FOR REPRESENTATIVE QUERIES (no threshold applied)")
    print("=" * 72)
    print_representative_queries(embedding_model, index, df)

    print("\n" + "=" * 72)
    print("4. PER-CASE DETAIL: rank / score / threshold survival, every RETRIEVE case")
    print("=" * 72)
    per_case_threshold_table(per_case_raw)

    print("\n" + "=" * 72)
    print("This script does not choose or recommend a threshold. Use the numbers")
    print("above as evidence before setting retrieval_pipeline.SIMILARITY_THRESHOLD.")
    print("=" * 72)


if __name__ == "__main__":
    main()
