from flask import Flask, request, render_template, jsonify
from fanout_ai import run_tool

app = Flask(
    __name__,
    # Note: Ensure this path is actually correct for your server environment
    template_folder="/home/eckoc1910/Cursor-Vibe-Coding-Repo/templates"
)

@app.route("/api/fanout", methods=["POST"])
def fanout():
    data = request.get_json() or {}
    content_text = data.get("content_text") or data.get("content") or ""
    keyword = data.get("keyword") or ""

    result = run_tool(content_text, keyword)
    return jsonify(result)

# --- FIXED INDEX ROUTE ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    # 1. Initialize variables explicitly so they exist for GET requests too
    submitted_keyword = ""
    submitted_text = ""

    if request.method == "POST":
        try:
            # 2. Capture form data into the variables we initialized
            submitted_text = request.form.get("content_text", "")
            submitted_keyword = request.form.get("keyword", "")

            # Run the tool
            output = run_tool(content_text=submitted_text, keyword=submitted_keyword)

            # Validate output
            if isinstance(output, dict) and "result" in output:
                result = output["result"]
            elif isinstance(output, dict) and "error" in output:
                error = output["error"]
            else:
                error = "Unexpected response format from AI tool."

        except Exception as e:
            error = f"An error occurred: {str(e)}"

    # 3. Render template with all variables safely defined
    return render_template(
        'index.html',
        result=result,
        error=error,
        submitted_keyword=submitted_keyword,
        submitted_text=submitted_text
    )

if __name__ == "__main__":
    app.run(debug=True)
