from flask import Flask, render_template, request

from fanout_ai import run_tool

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    content_text = ""
    keyword = ""

    if request.method == "POST":
        content_text = request.form.get("content_text", "")
        keyword = request.form.get("keyword", "")

        try:
            result_model = run_tool(content_text, keyword)
            result = result_model.model_dump()
        except Exception as exc:  # noqa: BLE001 - surfaced to user in UI
            error = str(exc)

    return render_template(
        "index.html",
        result=result,
        error=error,
        content_text=content_text,
        keyword=keyword,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
