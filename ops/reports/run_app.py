import os
import sys

from streamlit.web import cli as stcli


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

