import time
import requests
from typing import Dict, Any, Optional
from app.config import config
from app.logger import logger

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

class AuthManager:
    def __init__(self):
        self._access_token: Optional[str] = None
        self._expires_at: float = 0.0

    def get_access_token(self) -> str:
        """Returns a valid access token. Requests a new one if expired or not available."""
        # Refresh token if it's expired or expires within 60 seconds
        if not self._access_token or time.time() >= self._expires_at - 60:
            self._fetch_token()
        return self._access_token

    def _fetch_token(self) -> None:
        """Fetches a new access token using client credentials from OAuth2 endpoint."""
        config.validate()  # Ensure client_id and client_secret are loaded
        
        data = {
            "grant_type": "client_credentials",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
        }
        
        logger.info("Requesting new OAuth2 access token from Copernicus Data Space Ecosystem...")
        try:
            response = requests.post(TOKEN_URL, data=data, timeout=30)
            response.raise_for_status()
            token_data = response.json()
            
            self._access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 600)  # default to 10 mins if not provided
            self._expires_at = time.time() + expires_in
            logger.info("Successfully acquired access token.")
        except Exception as e:
            logger.error(f"Failed to authenticate with Copernicus Data Space Ecosystem: {e}")
            raise RuntimeError(f"Authentication failed: {e}") from e

    def get_auth_header(self) -> Dict[str, str]:
        """Convenience method to return headers for requests."""
        return {"Authorization": f"Bearer {self.get_access_token()}"}

# Shared authentication manager instance
auth_manager = AuthManager()
