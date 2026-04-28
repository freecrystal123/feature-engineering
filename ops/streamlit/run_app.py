import os
import sys

from streamlit.web import cli as stcli

'''
%sql
GRANT USE CATALOG ON CATALOG ml_prod TO `99573949-7059-437a-b0ba-c837cb609554`;
GRANT BROWSE ON CATALOG ml_prod TO `99573949-7059-437a-b0ba-c837cb609554`;
GRANT USE SCHEMA ON CATALOG ml_prod TO `99573949-7059-437a-b0ba-c837cb609554`;
GRANT READ VOLUME ON CATALOG ml_prod TO `99573949-7059-437a-b0ba-c837cb609554`;
'''

def main() -> None:
    port = os.getenv("DATABRICKS_APP_PORT", "8000")
    sys.argv = [
        "streamlit",
        "run",
        "app.py",
        "--server.address=0.0.0.0",
        f"--server.port={port}",
        "--server.headless=true",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()

