"""
Authentication Service Module
This module handles user authentication logic for the TeaCast API.
Currently implements a simple hardcoded authentication system.
In a production environment, this should be replaced with a secure authentication system.
"""

# A simple service to authenticate users (for now, hardcoded)
def authenticate_user(username, password):
    """
    Authenticate the user (admin or user).
    
    Args:
        username (str): The username to authenticate
        password (str): The password to verify
        
    Returns:
        dict: If authentication successful, returns {"username": username, "role": role}
        None: If authentication fails
        
    Note:
        This is a simplified authentication system for development purposes.
        In production, implement proper password hashing and database storage.
    """
    # Hardcoded user credentials (for development only)
    users = {
        'admin': {'password': 'admin', 'role': 'admin'},
        'user': {'password': 'user', 'role': 'user'}
    }

    # Get user from the users dictionary
    user = users.get(username)

    # Verify password and return user data if successful
    if user and user['password'] == password:
        return {"username": username, "role": user['role']}  # Return user with role
    else:
        return None  # Authentication failed
