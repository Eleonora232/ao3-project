from pathlib import Path

html_path = Path("/Users/ele/ao3-project2/docs/index.html")
with open(html_path, "r") as f:
    content = f.read()

# occurrences plot replacement
target_occurrences = """            <!-- Image placeholder for occurrences plot -->
            <div class="image-placeholder">
                <svg viewBox="0 0 24 24" stroke="currentColor" fill="none">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                <div class="placeholder-text">
                    <strong>Fandom Size Distribution Plot</strong><br>
                    Add a screenshot of the inclusive Matplotlib plot here (frequency up to 150 works per fandom).
                </div>
                <button class="placeholder-btn" onclick="alert('Copy your saved chart image here.')">Add Image</button>
            </div>"""

replacement_occurrences = """            <!-- Fandom distribution plot -->
            <div class="blog-image-container">
                <img src="fandom_distribution.png" alt="Fandom Size Distribution Plot" class="blog-image">
                <div class="image-caption">Impact of Occurrence Threshold on Fandom Tags and Works Distribution (cumulative range up to 150).</div>
            </div>"""

if target_occurrences in content:
    content = content.replace(target_occurrences, replacement_occurrences)
    print("Successfully replaced occurrences!")
else:
    print("Error: Target for occurrences not found in content!")
    # Let's print exactly the placeholder text
    idx = content.find("Image placeholder for occurrences plot")
    if idx != -1:
        print("Exact content found:")
        print(repr(content[idx:idx+600]))

with open(html_path, "w") as f:
    f.write(content)
