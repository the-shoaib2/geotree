# Sentinel-2 Imagery Downloader for Bangladesh

This production-grade Python application automatically queries, downloads, verifies, and logs Sentinel-2 Level-2A imagery covering Bangladesh from the Copernicus Data Space Ecosystem (CDSE).

## Features

- **Authentication:** OAuth2 client credentials grant with automated token refresh.
- **Bangladesh AOI:** Polygon-based search intersecting Bangladesh (supports MultiPolygon).
- **Search:** Query official Copernicus STAC API with cloud cover, date-ranges, and pagination.
- **Download Manager:** Resumes interrupted downloads, performs retries (5 attempts), and checks integrity.
- **Metadata Database:** Persistent SQLite storage tracking download metadata, checksums, and paths.
- **Reporting:** Automatic generation of JSON/CSV summary reports on download runs.
- **CLI Commands:** Multi-command interface (`search`, `download`, `resume`, `verify`, `list`).

## Folder Structure

```text
sentinel_downloader/
├── app/
│   ├── auth.py
│   ├── search.py
│   ├── downloader.py
│   ├── database.py
│   ├── logger.py
│   ├── config.py
│   ├── utils.py
│   ├── checksum.py
│   └── cli.py
├── data/
├── logs/
├── database/
├── config.yaml
├── .env
├── requirements.txt
├── README.md
└── main.py
```

## Setup & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure credentials:**
   Fill in your Client ID and Client Secret in the `.env` file:
   ```env
   CLIENT_ID=your_client_id
   CLIENT_SECRET=your_client_secret
   ```

3. **Configure Search Parameters:**
   Configure date range, cloud cover limit, and other parameters in `config.yaml`.

## Usage

- **Search for products:**
  ```bash
  python main.py search
  ```

- **Download matching products:**
  ```bash
  python main.py download
  ```

- **Download with a custom date range override:**
  ```bash
  python main.py download --start 2026-01-01 --end 2026-01-31
  ```

- **Resume downloads:**
  ```bash
  python main.py resume
  ```

- **Verify integrity of downloaded files:**
  ```bash
  python main.py verify
  ```

- **List status of all tracked products:**
  ```bash
  python main.py list
  ```
