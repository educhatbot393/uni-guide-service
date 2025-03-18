from flask import jsonify

def success_response(data=None, message="Success"):
    """Generate a standardized success response"""
    response = {
        "status": "success",
        "message": message
    }
    
    if data is not None:
        response["data"] = data
        
    return jsonify(response)

def error_response(message="An error occurred", status_code=500):
    """Generate a standardized error response"""
    response = {
        "status": "error",
        "message": message
    }
    
    return jsonify(response), status_code

def progress_response(progress, message="In progress"):
    """Generate a standardized progress response"""
    response = {
        "status": "in_progress",
        "message": message,
        "progress": progress
    }
    
    return jsonify(response)