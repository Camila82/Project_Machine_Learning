from flask import Blueprint, render_template

# Define the blueprint
business_bp = Blueprint('business', __name__)

@business_bp.route('/business-understanding')
def understanding():
    return render_template('business_understanding.html')