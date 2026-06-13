with open("/Users/ele/ao3-project2/docs/index.html", "r") as f:
    content = f.read()

print(f"File length: {len(content)}")
print(f"Contains 'blog-image-container': {'blog-image-container' in content}")
print(f"Contains 'fandom_distribution.png': {'fandom_distribution.png' in content}")
print(f"Contains 'marvel_teen_wolf_deep_dive.png': {'marvel_teen_wolf_deep_dive.png' in content}")
