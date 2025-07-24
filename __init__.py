from dotenv import load_dotenv
import os
from flask import Flask
from partfinal.config import Config  # Import from config.py
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_mail import Mail
from twilio.rest import Client

app = Flask(__name__)


app.config.from_object(Config)  # Load all settings from Config class

#Make this app availabe in the shell
ctx = app.app_context()
ctx.push()

app.config['DATASET_DIR'] = os.path.join(app.root_path, 'datasets')
# Ensure datasets directory exists
os.makedirs(app.config['DATASET_DIR'], exist_ok=True)


app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] ='postgresql://ishmael:test@localhost:5432/medicalapp'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'patient_login'
login_manager.login_message_category = 'info'

# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587  # TLS port
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = os.environ.get('EMAIL_USER')  # Gmail address from environment variable
app.config['MAIL_PASSWORD'] = os.environ.get('EMAIL_PASS')  # App-specific password or regular password
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('EMAIL_USER')  # Default sender is the same as username


# Initialize Twilio client
twilio_client = Client(
    app.config['TWILIO_ACCOUNT_SID'],
    app.config['TWILIO_AUTH_TOKEN']
)

mail = Mail(app)





from partfinal import routes