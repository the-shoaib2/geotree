#!/usr/bin/env python3
"""
Sentinel-2 level-2A Bangladesh Downloader Entry Point
"""
import sys
from app.cli import run_cli
from app.logger import logger

def main():
    try:
        run_cli()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unhandled application exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
