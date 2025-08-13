# ... [imports unchanged] ...

def main():
    ART_DIR.mkdir(exist_ok=True, parents=True)
    slide_snippets = []
    topics = random.sample(INTERESTS, k=min(NUM_ARTICLES, len(INTERESTS)))

    for i,topic in enumerate(topics, start=1):
        try:
            url = ask_url(topic)
            payload = write_article(url)

            queries = payload.get("image_queries") or [payload.get("title","news")]
            images = []
            for q in queries[:3]:
                img = unsplash(q)
                if img: images.append(img)
            if not images:
                images.append("assets/fallback-1.jpg")

            html_page = render_article(payload, images)
            art_path = ART_DIR / f"article{i}.html"
            with open(art_path, "w", encoding="utf-8") as f:
                f.write(html_page)

            title = payload.get("title") or first_sentence_html(payload.get("body_html",""))
            hero = images[0]
            style_bg = f" style=\"background-image:url('{hero}')\""
            slide = (
                f"<a class=\"slide\" href=\"articles/article{i}.html\"{style_bg}>"
                f"<span class=\"badge\">Top Story</span>"
                f"<h2>{title}</h2>"
                f"</a>"
            )
            slide_snippets.append(slide)

        except Exception as ex:
            print("Error generating article:", ex)

    if not slide_snippets:
        slide_snippets.append(
            "<a class=\"slide\" href=\"articles/article1.html\" style=\"background-image:url('assets/fallback-1.jpg')\">"
            "<span class=\"badge\">Update</span><h2>New stories will appear here shortly.</h2></a>"
        )

    update_index("\n".join(slide_snippets))
