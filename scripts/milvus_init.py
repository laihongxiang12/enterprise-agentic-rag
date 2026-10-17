"""Create or validate the configured Milvus collection."""

import asyncio

from app.core.config import Settings
from app.stores.milvus import MilvusVectorStore


async def main() -> None:
    settings = Settings()
    if settings.milvus_uri is None:
        raise SystemExit("MILVUS_URI is required")
    store = MilvusVectorStore(
        settings.milvus_uri,
        settings.milvus_collection,
        settings.embedding_dimension,
    )
    try:
        await store.initialize()
    finally:
        await store.close()
    print(
        f"Milvus collection {settings.milvus_collection!r} is ready "
        f"with dimension {settings.embedding_dimension}."
    )


if __name__ == "__main__":
    asyncio.run(main())
