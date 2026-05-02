import chromadb

client = chromadb.PersistentClient(path="D:/Mosaic/data/kb")
col = client.get_collection("mosaic_kb")
results = col.get(include=["metadatas"])
video_ids = sorted(set(m["video_id"] for m in results["metadatas"]))
print(f"Total entries: {col.count()}")
print(f"Unique videos analyzed: {len(video_ids)}")
for vid in video_ids:
    channel = next(m["channel"] for m in results["metadatas"] if m["video_id"] == vid)
    print(f"  {vid}  ({channel})")
