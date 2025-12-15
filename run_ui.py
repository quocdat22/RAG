"""
Run script for Streamlit UI.
Ensures proper Python path setup.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now run streamlit
if __name__ == "__main__":
    import streamlit.web.cli as stcli

    st_file = str(project_root / "ui" / "app.py")
    sys.argv = ["streamlit", "run", st_file]

    stcli.main()
