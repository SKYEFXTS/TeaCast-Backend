# A simple service to authenticate users (for now, hardcoded)
def authenticate_user(username, password):
    """Authenticate the user (admin or user)."""
    users = {
        'admin': {'password': 'admin', 'role': 'admin'},
        'user': {'password': 'user', 'role': 'user'}
    }

    user = users.get(username)

    if user and user['password'] == password:
        return {"username": username, "role": user['role']}  # Return user with role
    else:
        return None  # Authentication failed
