from flask import Flask, render_template, request

from fanout_ai import FanOutResult, run_tool

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result_data: FanOutResult | None = None
    error_message: str | None = None

    if request.method == "POST":
        content_text = request.form.get("content_text", "")
        keyword = request.form.get("keyword", "")

        response = run_tool(content_text, keyword)
        if "error" in response:
            error_message = response["error"]
        else:
            result_data = response.get("result")

    return render_template(
        "index.html",
        result=result_data,
        error=error_message,
        submitted_text=request.form.get("content_text", "") if request.method == "POST" else "",
        submitted_keyword=request.form.get("keyword", "") if request.method == "POST" else "",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
