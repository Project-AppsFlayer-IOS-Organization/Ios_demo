import subprocess
import json
import urllib.request


def run_security_scan(file_path):
    """Runs a static security scan on the file using Semgrep"""
    try:
        result = subprocess.run(
            ["semgrep", "--config=auto", file_path, "--json"],
            capture_output=True,
            text=True
        )
        scan_data = json.loads(result.stdout)
        results_summary = []
        for index, finding in enumerate(scan_data.get("results", [])):
            results_summary.append({
                "issue_number": index + 1,
                "line": finding.get("start", {}).get("line"),
                "message": finding.get("extra", {}).get("message"),
                "severity": finding.get("extra", {}).get("severity")
            })
        return json.dumps(results_summary, indent=2)
    except Exception as e:
        return f"Error running static scan: {str(e)}"


def run_ai_agent_review(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        code_to_review = f.read()

    static_report = run_security_scan(file_path)

    # הדביקי כאן את המפתח שיופיע לך מיד לאחר הלחיצה על Create
    api_key = os.getenv("OPENROUTER_API_KEY")

    url = "https://openrouter.ai/api/v1/chat/completions"

    system_instruction = (
        "You are an automated Code Quality and Security Agent (similar to SonarQube and Checkstyle).\n"
        "Analyze the code and the provided static analysis report.\n"
        "Identify: 1) Code style issues, naming conventions, and code smells. 2) Security vulnerabilities.\n"
        "Provide your output ONLY as a valid JSON list of objects. Each object must have:\n"
        "- 'category': 'style' or 'security'\n"
        "- 'line': line number\n"
        "- 'description': explain the problem\n"
        "- 'suggested_fix': provide the exact corrected code snippet"
    )

    user_prompt = f"Code to analyze:\n\n{code_to_review}\n\nStatic scan tool report:\n{static_report}"

    payload = {
        "model": "meta-llama/llama-3-8b-instruct:free",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    try:
        json_data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, headers=headers, method="POST")

        with urllib.request.urlopen(req) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            return response_data['choices'][0]['message']['content']
    except Exception as e:
        return f"Error in AI call: {str(e)}"


if __name__ == "__main__":
    test_file = "test_code.py"

    print("Starting secure review via OpenRouter... Please wait...")
    final_report = run_ai_agent_review(test_file)

    print("\n--- Final Agent Report (JSON) ---")
    print(final_report)