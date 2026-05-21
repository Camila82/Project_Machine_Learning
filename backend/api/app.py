from flask import Flask, render_template
import webbrowser
import threading
import os

# Configuramos Flask para que busque el HTML y los estáticos en la carpeta frontend
app = Flask(__name__, 
            template_folder="../../frontend", 
            static_folder="../../frontend",
            static_url_path="")

@app.route('/')
def home():
    # Flask buscará 'index.html' dentro de la ruta definida en template_folder
    return render_template('index.html')

def open_browser():
    # Abrimos el navegador en la dirección local
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # Usamos un hilo para que no bloquee el inicio del servidor
    threading.Timer(1.5, open_browser).start()
    
    # Importante: debug=False evita que el navegador se abra dos veces
    app.run(port=5000, debug=False)

