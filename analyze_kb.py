import chromadb
import json
from collections import defaultdict

client = chromadb.PersistentClient(path="D:/Mosaic/data/kb")
col = client.get_collection("mosaic_kb")
results = col.get(include=["metadatas", "documents"])

# Group by video
videos = defaultdict(lambda: {"channel": "", "title": "", "insights": [], "meta": {}})
for doc, meta in zip(results["documents"], results["metadatas"]):
    vid = meta["video_id"]
    if vid == "__seed__":
        continue
    videos[vid]["channel"] = meta["channel"]
    videos[vid]["title"] = meta.get("title", "")
    videos[vid]["insights"].append(doc)
    if meta.get("full_json"):
        videos[vid]["meta"] = json.loads(meta["full_json"])

print(f"=== KB INTELLIGENCE REPORT ===")
print(f"Total videos: {len(videos)}")
print(f"Total insight entries: {col.count()}")
print()

# By channel
by_channel = defaultdict(list)
for vid, data in videos.items():
    by_channel[data["channel"]].append(data)

print("=== COVERAGE BY CHANNEL ===")
for channel, vids in sorted(by_channel.items()):
    print(f"  {channel}: {len(vids)} videos, {sum(len(v['insights']) for v in vids)} insights")
print()

# Hook structures
print("=== HOOK TYPES FOUND ===")
hooks = defaultdict(int)
for vid, data in videos.items():
    h = data["meta"].get("hook", {}).get("structure", "unknown")
    hooks[h] += 1
for h, count in sorted(hooks.items(), key=lambda x: -x[1]):
    print(f"  {count}x  {h}")
print()

# Arc shapes
print("=== STORY ARC SHAPES ===")
arcs = defaultdict(int)
for vid, data in videos.items():
    a = data["meta"].get("story_arc", {}).get("shape", "unknown")
    arcs[a] += 1
for a, count in sorted(arcs.items(), key=lambda x: -x[1]):
    print(f"  {count}x  {a}")
print()

# Narration styles
print("=== NARRATION RHYTHM PATTERNS ===")
rhythms = defaultdict(int)
for vid, data in videos.items():
    r = data["meta"].get("narration_style", {}).get("rhythm", "unknown")
    rhythms[r] += 1
for r, count in sorted(rhythms.items(), key=lambda x: -x[1]):
    print(f"  {count}x  {r}")
print()

# Sample insights from each channel
print("=== SAMPLE KEY INSIGHTS (2 per channel) ===")
for channel, vids in sorted(by_channel.items()):
    print(f"\n-- {channel} --")
    for vid in vids[:2]:
        print(f"  [{vid['title'][:50]}]")
        for ins in vid["insights"][:2]:
            print(f"    • {ins[:120]}")
