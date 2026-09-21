import os
import sys

print("Running web_report.py...")
api_key = os.environ.get("JQUANTS_API_KEY")
if not api_key:
    print("Error: 環境変数 JQUANTS_API_KEY が設定されていません。")
    sys.exit(1)
print("JQUANTS_API_KEY is set correctly.")
