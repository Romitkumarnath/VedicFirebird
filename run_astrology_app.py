#!/usr/bin/env python3
"""
Vedic Astrology Bot Launcher
----------------------------
This script launches the Vedic Astrology Bot GUI application.
"""

import os
import sys
import traceback
from astrology_gui import create_interface

def main():
    try:
        # Create and launch the interface
        interface = create_interface()
        interface.launch(
            server_name="127.0.0.1",  # Run locally only
            share=False,
            inbrowser=True,  # Automatically open in browser
            show_api=False,  # Don't show API
        )
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main() 