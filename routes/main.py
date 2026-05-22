from flask import Blueprint, render_template

# Define the blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/methodology')
def methodology():
    return render_template('methodology.html')