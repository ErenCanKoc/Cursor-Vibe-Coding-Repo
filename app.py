from flask import Flask, render_template, request

from fanout_ai import FanOutResult, run_tool

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result_data: FanOutResult | None = None
    error_message: str | None = None
    combined_content = ""

    if request.method == "POST":
        content_text = request.form.get("content_text", "")
        keyword = request.form.get("keyword", "")
        uploaded_file = request.files.get("content_file")

        uploaded_text = ""
        if uploaded_file and uploaded_file.filename:
            try:
                uploaded_text = uploaded_file.read().decode("utf-8").strip()
            except UnicodeDecodeError:
                error_message = "Uploaded file must be UTF-8 encoded text."
            except Exception:
                error_message = "Unable to read the uploaded file. Please try again."

        combined_content_parts = [part for part in [content_text.strip(), uploaded_text] if part]
        combined_content = "\n\n".join(combined_content_parts)

        if error_message is None:
            response = run_tool(combined_content, keyword)
            if "error" in response:
                error_message = response["error"]
            else:
                result_data = response.get("result")

    return render_template(
        "index.html",
        result=result_data,
        error=error_message,
        submitted_text=combined_content if request.method == "POST" else "",
        submitted_keyword=request.form.get("keyword", "") if request.method == "POST" else "",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
