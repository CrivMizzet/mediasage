#!/usr/bin/env python3
"""
Simple Jellyfin Authentication Script
Connects to Jellyfin server, authenticates, and stores the access token.
Supports dynamic configuration via command line arguments, environment variables, or interactive input.
"""

import requests
import json
import os
import argparse
import getpass
from urllib.parse import urljoin

class JellyfinAuth:
    def __init__(self, server_url, username, password):
        """
        Initialize Jellyfin authentication client
        
        Args:
            server_url (str): Full URL to Jellyfin server (e.g., 'http://localhost:8096')
            username (str): Jellyfin username
            password (str): Jellyfin password
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.access_token = None
        self.user_id = None
        
        # Client identification (required by Jellyfin API)
        self.client_name = "PythonJellyfinClient"
        self.client_version = "1.0.0"
        self.device_name = "Python Script"
        self.device_id = "python-jellyfin-auth"
        
    def authenticate(self):
        """
        Authenticate with Jellyfin server using username/password
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        # Construct authentication URL
        auth_url = urljoin(self.server_url, '/Users/AuthenticateByName')
        
        # Prepare headers as per Jellyfin API documentation
        headers = {
            'Content-Type': 'application/json',
            'X-Emby-Authorization': (
                f'MediaBrowser Client="{self.client_name}", '
                f'Device="{self.device_name}", '
                f'DeviceId="{self.device_id}", '
                f'Version="{self.client_version}"'
            )
        }
        
        # Prepare authentication payload
        auth_data = {
            'Username': self.username,
            'Pw': self.password
        }
        
        try:
            # Send authentication request
            response = requests.post(
                auth_url,
                headers=headers,
                json=auth_data,
                timeout=10
            )
            
            # Check if authentication was successful
            if response.status_code == 200:
                auth_response = response.json()
                
                # Extract access token and user ID from response
                self.access_token = auth_response.get('AccessToken')
                self.user_id = auth_response.get('User', {}).get('Id')
                
                if self.access_token:
                    print(f"✓ Authentication successful for user: {self.username}")
                    print(f"✓ User ID: {self.user_id}")
                    return True
                else:
                    print("✗ Authentication failed: No access token received")
                    return False
                    
            else:
                print(f"✗ Authentication failed: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            return False
    
    def save_token(self, filepath='jellyfin_token.json'):
        """
        Save authentication token to file
        
        Args:
            filepath (str): Path to save token file
        """
        if not self.access_token:
            print("✗ No authentication token to save")
            return False
            
        token_data = {
            'server_url': self.server_url,
            'username': self.username,
            'user_id': self.user_id,
            'access_token': self.access_token,
            'client_name': self.client_name,
            'device_id': self.device_id
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(token_data, f, indent=2)
            print(f"✓ Token saved to: {filepath}")
            return True
        except IOError as e:
            print(f"✗ Failed to save token: {e}")
            return False
    
    def load_token(self, filepath='jellyfin_token.json'):
        """
        Load authentication token from file
        
        Args:
            filepath (str): Path to token file
            
        Returns:
            bool: True if token loaded successfully, False otherwise
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    token_data = json.load(f)
                
                self.access_token = token_data.get('access_token')
                self.user_id = token_data.get('user_id')
                
                if self.access_token:
                    print(f"✓ Token loaded from: {filepath}")
                    return True
                    
            print(f"✗ No valid token found in: {filepath}")
            return False
            
        except (IOError, json.JSONDecodeError) as e:
            print(f"✗ Failed to load token: {e}")
            return False

def get_credentials():
    """
    Get Jellyfin server credentials from multiple sources (in order of priority):
    1. Command line arguments
    2. Environment variables
    3. Interactive input
    
    Returns:
        tuple: (server_url, username, password)
    """
    parser = argparse.ArgumentParser(description='Jellyfin Authentication Script')
    parser.add_argument('--server', '-s', help='Jellyfin server URL (e.g., http://localhost:8096)')
    parser.add_argument('--username', '-u', help='Jellyfin username')
    parser.add_argument('--password', '-p', help='Jellyfin password')
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive input')
    
    args = parser.parse_args()
    
    # Get values from command line args, environment variables, or interactive input
    server_url = args.server or os.getenv('JELLYFIN_SERVER_URL')
    username = args.username or os.getenv('JELLYFIN_USERNAME')
    password = args.password or os.getenv('JELLYFIN_PASSWORD')
    
    # If any value is missing or interactive mode is requested, prompt for input
    if not server_url or not username or not password or args.interactive:
        print("Enter Jellyfin server credentials:")
        
        if not server_url:
            server_url = input("Server URL (e.g., http://localhost:8096): ").strip()
            
        if not username:
            username = input("Username: ").strip()
            
        if not password:
            password = getpass.getpass("Password: ")
    
    # Validate inputs
    if not server_url or not username or not password:
        raise ValueError("Server URL, username, and password are all required")
    
    # Ensure server URL has protocol
    if not server_url.startswith(('http://', 'https://')):
        server_url = f"http://{server_url}"
    
    return server_url, username, password

def create_jellyfin_client(server_url=None, username=None, password=None):
    """
    Create a JellyfinAuth client with the provided credentials.
    If credentials are not provided, they will be obtained dynamically.
    
    Args:
        server_url (str, optional): Jellyfin server URL
        username (str, optional): Jellyfin username  
        password (str, optional): Jellyfin password
        
    Returns:
        JellyfinAuth: Configured Jellyfin authentication client
    """
    if not all([server_url, username, password]):
        # If any credentials are missing, get them dynamically
        server_url, username, password = get_credentials()
    
    return JellyfinAuth(server_url, username, password)

def main():
    """
    Main function to demonstrate Jellyfin authentication
    """
    print("Jellyfin Authentication Script")
    print("=" * 40)
    
    try:
        # Get credentials dynamically
        server_url, username, password = get_credentials()
        
        # Create authentication client
        jellyfin = JellyfinAuth(server_url, username, password)
        
        # Try to load existing token first
        if jellyfin.load_token():
            print("Using existing authentication token")
        else:
            # Authenticate with server
            print(f"Connecting to Jellyfin server: {server_url}")
            
            if jellyfin.authenticate():
                # Save the token for future use
                jellyfin.save_token()
            else:
                print("Authentication failed. Please check your credentials and server URL.")
                return
        
        print(f"\nAccess Token: {jellyfin.access_token}")
        print("\nAuthentication completed successfully!")
        
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    except KeyboardInterrupt:
        print("\n✗ Operation cancelled by user")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

if __name__ == "__main__":
    main()