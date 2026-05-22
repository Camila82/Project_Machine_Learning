from flask import Flask

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev_secret_key_urban_noise'

    # Register Blueprints
    from routes.main import main_bp
    from routes.business import business_bp
    from routes.data import data_bp
    from routes.rl_agent import rl_bp  # <--- NUEVO MODULO FASE 2
    
    app.register_blueprint(main_bp)
    app.register_blueprint(business_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(rl_bp)      # <--- NUEVO MODULO FASE 2

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)