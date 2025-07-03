from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username'].strip().lower()
        email = request.form['email'].strip()

        # Save user session or pass via redirect
        return redirect(url_for('voice_interface', username=username))
    return render_template('index.html')


@app.route('/voice/<username>', methods=['GET'])
def voice_interface(username):
    model_path = f"models/{username}_voice_model.pkl"
    model_exists = os.path.exists(model_path)
    return render_template('voice_interface.html', username=username, model_exists=model_exists)


if __name__ == '__main__':
    app.run(debug=True)
