from rag import search


def search_knowledge(
    query,
    index,
    chunks,
    top_k=5,
):
    results = search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    if not results:
        return {
            "answerable": False,
            "sources": [],
        }

    return {
        "answerable": True,
        "sources": [
            {
                "text": result["text"],
                "source": result["source"],
                "page": result["page"],
                "score": result["score"],
            }
            for result in results
        ],
    }