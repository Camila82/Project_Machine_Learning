from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev_secret_key_urban_noise'

    from routes.main import main_bp
    from routes.business import business_bp
    from routes.data import data_bp
    from routes.rl_agent import rl_bp
    from routes.models_bp import models_bp # <--- NUEVO
    
    app.register_blueprint(main_bp)
    app.register_blueprint(business_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(rl_bp)
    app.register_blueprint(models_bp)      # <--- NUEVO

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)