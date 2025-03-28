"""
Authentication Controller Module
This module handles user authentication and authorization for the TeaCast API.
It provides endpoints for user login and manages user sessions.
"""

from flask import Blueprint, request, jsonify, flash
from Service.authService import authenticate_user

# Create Blueprint for the auth route
# This blueprint will handle all authentication-related endpoints
auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/login', methods=['POST'])
def login():
    """
    Handles login requests for users and admin.
    
    Expected JSON payload:
    {
        "username": "string",
        "password": "string"
    }
    
    Returns:
        JSON response with:
        - Success (200): {"message": "role logged in successfully", "role": "string"}
        - Bad Request (400): {"error": "Invalid credentials"}
        - Server Error (500): {"error": "error message"}
    """
    try:
        # Get the data from the request
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        # Authenticate the user using the auth service
        user = authenticate_user(username, password)

        if user:
            # If authentication is successful, return the user role
            return jsonify({'message': f'{user["role"]} logged in successfully', 'role': user["role"]}), 200
        else:
            # If authentication fails, return error message
            return jsonify({'error': 'Invalid credentials'}), 400
    except Exception as e:
        # Handle any unexpected errors during authentication
        return jsonify({'error': str(e)}), 500
