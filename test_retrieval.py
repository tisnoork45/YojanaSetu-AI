
import retrieval_pipeline as rp

def run_query(query, embedding_model, index, df, profile=None):
    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    candidates = rp.retrieve_candidates(query, embedding_model, index, df)
    if not candidates:
        print(rp.NO_MATCH_MESSAGE)
        return

    rp.attach_eligibility(candidates, profile or {})
    recommended, not_eligible = rp.rank_candidates(candidates)

    for c in recommended:
        print(f"\n{c['scheme_name']}  (similarity={c['_similarity']:.2f}, eligibility={c['_eligibility'].status})")
        print(f"  Category: {c['category']}")
        print(f"  {c['_eligibility'].summary}")

    if not_eligible:
        print("\n-- relevant but not eligible --")
        for c in not_eligible:
            print(f"{c['scheme_name']}: {c['_eligibility'].summary}")


if __name__ == "__main__":
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

    run_query("I need money for my crops", embedding_model, index, df,
              profile={"occupation": "farmer"})
    run_query("I'm a 65 year old farmer looking for a pension", embedding_model, index, df,
              profile={"age": 65, "occupation": "farmer"})
    run_query("What's the weather like today?", embedding_model, index, df)
