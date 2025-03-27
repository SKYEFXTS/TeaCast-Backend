from flask import Blueprint, request, jsonify, flash
from Service.authService import authenticate_user

# Create Blueprint for the auth route
auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/login', methods=['POST'])
def login():
    """Handles login requests for users and admin."""
    try:
        # Get the data from the request
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        # Authenticate the user
        user = authenticate_user(username, password)

        if user:
            # If authentication is successful, return the user role
            return jsonify({'message': f'{user["role"]} logged in successfully', 'role': user["role"]}), 200
        else:
            # If authentication fails
            return jsonify({'error': 'Invalid credentials'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
