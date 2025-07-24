from partfinal import app, db, bcrypt, mail
from partfinal.database_models import User, Appointment, Prescription, PatientMessage
from flask import render_template, url_for, flash, session, jsonify
#create a route for the login page
#from  partfinal.forms import RequestResetForm, LoginForm #RegistrationForm, LoginForm, RequestResetForm, ResetPasswordForm
from partfinal.forms import RequestResetForm, PatientLoginForm, DoctorLoginForm, AdminLoginForm, AppointmentForm, PatientMessageForm, EditPatientForm, ResetPasswordForm, ChangePasswordForm, PrescriptionForm
from flask import Flask, render_template, request, send_file, redirect, current_app, send_file, jsonify,make_response
import numpy as np
import pandas as pd
import pickle
from io import BytesIO
import csv
from fpdf import FPDF
import os
import ast
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
import smtplib
from fpdf import FPDF
import xlsxwriter
from werkzeug.utils import secure_filename
from sqlalchemy import func, distinct
from datetime import datetime, date
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VideoGrant
from twilio.rest import Client
from flask import abort
from sqlalchemy import func, distinct, or_, and_ 



# for data analysis page
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import humanize





app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'

app.config['DATASET_DIR'] = os.path.join(app.root_path, 'datasets')
# Later in your code
dataset_path = os.path.join(app.config['DATASET_DIR'], 'Training.csv')
df = pd.read_csv(dataset_path)


# LOAD DATASETS
import os
import pandas as pd
from pathlib import Path

# Get the directory where routes.py is located
current_dir = Path(__file__).parent

# Construct the path to your CSV
dataset_path = current_dir.parent / 'datasets' / 'symtoms.csv'  # Note parent for going up one level
symptoms = pd.read_csv("C:/Users/hp/OneDrive/Desktop/medical_dataset/symtoms_df.csv")


precautions = pd.read_csv("C:/Users/hp/OneDrive/Desktop/medical_dataset/precautions_df.csv")
descriptions = pd.read_csv("C:/Users/hp/OneDrive/Desktop/medical_dataset/description.csv")
workout = pd.read_csv("C:/Users/hp/OneDrive/Desktop/medical_dataset/workout_df.csv")
medications = pd.read_csv("C:/Users/hp/OneDrive/Desktop/medical_dataset/medications.csv")
diets = pd.read_csv("C:/Users/hp/OneDrive/Desktop/medical_dataset/diets.csv")


# load the model
svc = pickle.load(open("C:/Users/hp/OneDrive/Desktop/medical_dataset/svc.pkl", 'rb'))
# displaying predicted results(helper function) and prediction function
def helper(disease):
    desc = descriptions[descriptions['Disease'] == disease]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == disease]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == disease]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == disease] ['workout']


    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

#tesing page numaber 2
@app.route('/testing2')
def testing2():
    page = request.args.get('page', 1, type=int)

    # Pagination: 10 entries per page
    per_page = 10
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    # Slice the dataset for the current page
    page_data = df.iloc[start_idx:end_idx]

    # Convert the data to HTML for rendering
    table_html = page_data.to_html(classes="data", index=False)

    # Pagination links (next and previous)
    next_page = page + 1 if end_idx < len(df) else None
    prev_page = page - 1 if page > 1 else None

    return render_template('testing2.html', table=table_html, next_page=next_page, prev_page=prev_page, page=page)




@app.route('/download_csv')
def download_csv():
    # Save dataframe as CSV to send it
    csv_output = BytesIO()
    df.to_csv(csv_output, index=False)
    csv_output.seek(0)
    return send_file(csv_output, mimetype='text/csv', as_attachment=True, download_name='dataset.csv')

@app.route('/download_excel')
def download_excel():
    excel_output = BytesIO()
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_output.seek(0)
    return send_file(excel_output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='dataset.xlsx')

@app.route('/download_pdf')
def download_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    # Create PDF table with column headers
    for col in df.columns:
        pdf.cell(40, 10, col, border=1, align='C')
    pdf.ln()

    # Create PDF rows with data
    for _, row in df.iterrows():
        for val in row:
            pdf.cell(40, 10, str(val), border=1, align='C')
        pdf.ln()

    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    
    return send_file(pdf_output, mimetype='application/pdf', as_attachment=True, download_name='dataset.pdf')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['search_query']
    result = df[df.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
    return render_template('index.html', table=result.to_html(classes='data', header="true"))





@app.route('/')
@app.route('/index')
def index():
    return render_template('/index.html', posts={})


@app.route('/about')
def about():
    return render_template('/about.html')


@app.route('/services')
def services():
    return render_template('/services.html')


@app.route('/single-services')
def singleServices():
    return render_template('single-services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/solution')
def solution():
    return render_template('solution.html')

@app.route('/solution-details')
def solutionDetails():
    return render_template('solution-details.html')

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message(
        'Password Reset Request',
        sender=current_app.config['MAIL_DEFAULT_SENDER'],  # Uses configured sender
        recipients=[user.email]
    )
    msg.body = f'''To reset your password, visit:
{url_for('reset_token', token=token, _external=True)}

If you didn't request this, please ignore this email.
'''
    try:
        mail.send(msg)
        return True
    except Exception as e:
        current_app.logger.error(f"Email failed: {str(e)}")
        return False
        
@app.route('/reset_password', methods=['GET', 'POST'])
def request_reset_password():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
            flash('Password reset email sent. Check your inbox.', 'info')
            return redirect(url_for('signIn'))
    
    return render_template('reset_request.html', form=form)

#@app.route('/reset_password/<token>', methods=['GET', 'POST'])
#def reset_token(token):
    #if current_user.is_authenticated:
       # return redirect(url_for('index'))
    #user = User.verify_reset_token(token)
    #if user is None:
        #flash('That is an invalid or expired token', 'warning')
        #return redirect(url_for('request_reset_password'))
    #form = ResetPasswordForm()
    #if current_user.is_authenticated:
        #return redirect(url_for('dashboard'))
    #if form.validate_on_submit():
        #hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        #user.password = hashed_password
        #db.session.commit()
        #flash(f"Your password has been updated!. You are now able to login", "success")
        #return redirect(url_for('signIn'))
    #return render_template('/reset_token.html', title='Reset Password', form=form)
########################################################################################################################
########################################################################################################################
######################## BLOG ROUTES ###################################################################################

@app.route('/blog-1')
def blog1():
    return render_template('/blog-1.html')

@app.route('/blog-2')
def blog2():
    return render_template('/blog-2.html')

@app.route('/single-details')
def singleDetails():
    return render_template('single-blog.html')

########################################################################################################################
##########################################################################################################################
@app.route('/error-404')
def error404():
    return render_template('error-404.html')

@app.route('/sign-up', methods=['GET', 'POST'])
def signUp():
    pass
    #form = RegistrationForm()
    #if current_user.is_authenticated:
        #return redirect(url_for('dashboard'))
    #if form.validate_on_submit():
        #hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        #user = User(first_name=form.first_name.data, surname=form.surname.data, username=form.username.data,
                    #email=form.email.data, date_of_birth=form.date_of_birth.data, gender=form.gender.data,
                    #phone_number=form.phone_number.data, role=form.role.data, password_hash=hashed_password)
        #db.session.add(user)
        #db.session.commit()
        #flash(f"Your account has now created!. You are now able to login", "success")
        #return redirect(url_for('signIn'))
    #return render_template('/sign-up.html', title= 'Register', form=form)

@app.route('/sign-in', methods=['GET', 'POST'])
def signIn():
    pass
    #if current_user.is_authenticated:
        #return redirect(url_for('dashboard'))
    #form = LoginForm()
    #if form.validate_on_submit():
        #user = User.query.filter_by(username=form.username.data).first()
        #if user and bcrypt.check_password_hash(user.password_hash, form.password.data):
            #login_user(user, remember=form.remember.data)
            #next_page = request.args.get('next')
            #return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        
        #else:
            #flash('Login Unsuccessful. Please check username and password', 'danger')
    #return render_template('/sign-in.html', title= 'Login', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/appointment')
@login_required
def appointment():
    if not current_user.is_authenticated:
        return redirect(url_for('patient_login'))
    return render_template('appointment.html')


@app.route('/shop')
def shop():
    return render_template('shop.html')

@app.route('/cart')
def cart():
    return render_template('/cart.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/single-product')
def singleProduct():
    return render_template('single-product.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/privacy-policy')
def privacyPolicy():
    return render_template('privacy-policy.html')

###########################################################
#route for predictions page
# patients sending their challenges
###########################################################
@app.route('/predictions')
@login_required
def predictions():
    form = PatientMessageForm() 
    

    return render_template('predictions2.html', form = form) 

#route to habdle key error
@app.errorhandler(KeyError)
def handle_key_error(error):
    flash("⚠️ Oops! It looks like some of the symptoms you entered are not valid. Please double-check your entries and try again.", "danger")
    return redirect(url_for('predictions'))


#route for prediction form
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():

    form = PatientMessageForm() 

    # Ensure that current_user is logged in and valid
    if not current_user.is_authenticated:
        return redirect(url_for('patient_signup'))

    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if not symptoms:
            flash('⚠️ Please enter symptoms before submitting!', 'error')  # 'error' is the category
            return redirect(url_for('predict'))  # Redirect back to the form page
    
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [sym.strip("[]' ") for sym in user_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)
        desc,pre,med,die,wrkout = helper(predicted_disease)

        precautions = []
        for i in pre[0]:
            precautions.append(i)
        

        return render_template('predictions2.html', predicted_disease=predicted_disease, descriptions=desc, precautions=precautions, medications=med, diets=die, workout=wrkout, form=form)


@app.route('/tefri')
def tefri():
    return render_template('test_predict.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

#This route is used to display the dataset in a table format
@app.route('/testing')
@login_required
def testing():
    return render_template('testing.html', table=df.to_html(classes='data', header="true"))


@app.route('/layout')
@login_required
def layout():
    title = 'Artificial Intelligence-Powered Medical Recommender System with Telemedicine Support'
    return render_template('layout.html', title = title)

#it shows all the diseases available in our dataset
@app.route('/diseases')
@login_required
def diseases():
    return render_template('diseases.html')

# Custom 404 error handler
@app.errorhandler(404)
def page_not_found(error):
    return render_template('error-404.html')



@app.route('/data_analysis')
@login_required
def data_analysis():
    # Load your dataset (replace with your actual data loading code)
    #dataset_path = os.path.join('datasets', 'Training.csv')  # File in 'data' subfolder
    #dataset = pd.read_csv(dataset_path)
    
    # Option 2: Robust path (works even if script runs from another directory)
    current_dir = os.path.dirname(__file__)  # Gets the folder where your Python script lives
    dataset_path = os.path.join(current_dir, 'datasets', 'Training.csv')
    dataset = pd.read_csv(dataset_path)

    
    # Prepare data for template
    dataset_head = dataset.head()
    dataset_shape = dataset.shape
    missing_values = dataset.isnull().sum().sum()
    diseases = dataset['prognosis'].unique().tolist()
    
    # Assuming you have these metrics from your model training
    accuracy = 0.9088
    precision = 0.92
    recall = 0.89
    
    # Classification report (simplified example)
    # In practice, you'd use sklearn's classification_report output
    classification_rep = {
        "Fungal infection": {"precision": 1.0, "recall": 0.4, "f1_score": 0.57, "support": 5},
        "Allergy": {"precision": 1.0, "recall": 0.8, "f1_score": 0.89, "support": 5},
        # Add all other diseases...
    }

     # Replace this with your actual y_true and y_pred if available
    num_classes = len(diseases)
    y_true = np.random.randint(0, num_classes, 100)  # Random true labels
    y_pred = np.random.randint(0, num_classes, 100)  # Random predictions
    
    # Generate confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot to bytes
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    # Example lists for insights
    top_performers = ["Common Cold", "Diabetes", "Hepatitis E", "Malaria", "Migraine"]
    needs_improvement = ["AIDS", "Acne", "Dimorphic hemmorhoids(piles)", "Fungal infection", "Impetigo"]
    
    return render_template('data_analysis.html',
                         dataset_head=dataset_head,
                         dataset_shape=dataset_shape,
                         missing_values=missing_values,
                         diseases=diseases,
                         accuracy=accuracy,
                         precision=precision,
                         recall=recall,
                         classification_report=classification_rep,
                         top_performers=top_performers,
                         needs_improvement=needs_improvement)






# In routes.py
from partfinal.forms import (PatientRegistrationForm, DoctorRegistrationForm, 
                            AdminRegistrationForm, AppointmentForm)

@app.route('/signup/patient', methods=['GET', 'POST'])
def patient_signup():
    form = PatientRegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            first_name=form.first_name.data,
            surname=form.surname.data,
            username=form.username.data,
            email=form.email.data,
            date_of_birth=form.date_of_birth.data,
            gender=form.gender.data,
            phone_number=form.phone_number.data,
            password_hash=hashed_password,
            role='patient'
        )
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You can now log in', 'success')
        return redirect(url_for('patient_login'))
    return render_template('signup_patient.html', form=form)

@app.route('/signup/doctor', methods=['GET', 'POST'])
def doctor_signup():
    form = DoctorRegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            first_name=form.first_name.data,
            surname=form.surname.data,
            username=form.username.data,
            email=form.email.data,
            date_of_birth=form.date_of_birth.data,
            gender=form.gender.data,
            phone_number=form.phone_number.data,
            password_hash=hashed_password,
            role='doctor',
            specialization=form.specialization.data,
            license_number=form.license_number.data,
            hospital_affiliation=form.hospital_affiliation.data,
            is_approved=False  # Needs admin approval
        )
        db.session.add(user)
        db.session.commit()
        flash('Your doctor account has been created! Please wait for admin approval', 'success')
        return redirect(url_for('doctor_login'))
    return render_template('signup_doctor.html', form=form)

@app.route('/signup/admin', methods=['GET', 'POST'])
def admin_signup():
    form = AdminRegistrationForm()
    if form.validate_on_submit():
        if form.admin_code.data != 'YOUR_SECRET_ADMIN_CODE':  # Change this!
            flash('Invalid admin code', 'danger')
            return redirect(url_for('admin_signup'))
            
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(
            first_name=form.first_name.data,
            surname=form.surname.data,
            username=form.username.data,
            email=form.email.data,
            date_of_birth=form.date_of_birth.data,
            gender=form.gender.data,
            phone_number=form.phone_number.data,
            password_hash=hashed_password,
            role='admin',
            admin_code=form.admin_code.data
        )
        db.session.add(user)
        db.session.commit()
        flash('Admin account created successfully!', 'success')
        return redirect(url_for('admin_login'))
    return render_template('signup_admin.html', form=form)


# In routes.py
@app.route('/login/patient', methods=['GET', 'POST'])
def patient_login():
    if current_user.is_authenticated:
        if current_user.role == 'patient':
            return redirect(url_for('patient_dashboard'))
        else:
            logout_user()
            
    form = PatientLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.role == 'patient' and bcrypt.check_password_hash(user.password_hash, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('patient_dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login_patient.html', form=form)

@app.route('/login/doctor', methods=['GET', 'POST'])
def doctor_login():
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            logout_user()
            
    form = DoctorLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.role == 'doctor' and bcrypt.check_password_hash(user.password_hash, form.password.data):
            if not user.is_approved:
                flash('Your account is pending admin approval', 'warning')
                return redirect(url_for('doctor_login'))
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('doctor_dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login_doctor.html', form=form)

@app.route('/login/admin', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            logout_user()
            
    form = AdminLoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.role == 'admin' and bcrypt.check_password_hash(user.password_hash, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('admin_dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login_admin.html', form=form)

##############################################################################################################
# Patient Dashboard
##############################################################################################################
@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    if current_user.role != 'patient':
        flash('You are not authorized to view this page', 'danger')
        return redirect(url_for('index'))
    
    # Get upcoming approved appointments (status='Approved' and future dates)
    upcoming_appointments = Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.status == 'Approved',
        Appointment.appointment_date > datetime.utcnow()
    ).order_by(Appointment.appointment_date.asc()).all()
    
    # Get pending appointments
    pending_appointments = Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.status == 'Pending'
    ).order_by(Appointment.appointment_date.asc()).all()
    
    return render_template('patient_dashboard.html',
                         title='Patient Dashboard',
                         upcoming_appointments=upcoming_appointments,
                         pending_appointments=pending_appointments)


###################################################################################################################################
# Doctor Dashboard
###################################################################################################################################

def is_today(dt):
    return dt.date() == datetime.utcnow().date()

def is_soon(dt):
    return dt > datetime.utcnow() and (dt - datetime.utcnow()).total_seconds() < 86400  # Within 24 hours

def calculate_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor' or not current_user.is_approved:
        flash('You are not authorized to view this page', 'danger')
        return redirect(url_for('index'))
    
     # In your route
    today_start = datetime.combine(date.today(), datetime.min.time())
    today_end = datetime.combine(date.today(), datetime.max.time())

    todays_stats = db.session.query(
    Appointment.status,
    func.count(Appointment.id)
).filter(
    Appointment.doctor_id == current_user.id,
    Appointment.appointment_date.between(today_start, today_end)
).group_by(Appointment.status).all()


    pending_appointments = Appointment.query.filter_by(
        doctor_id=current_user.id,
        status='Pending'
    ).order_by(Appointment.appointment_date.asc()).all()
    
    upcoming_appointments = Appointment.query.filter_by(
        doctor_id=current_user.id,
        status='Approved'
    ).filter(
        Appointment.appointment_date >= datetime.utcnow()
    ).order_by(Appointment.appointment_date.asc()).all()
    
    past_appointments = Appointment.query.filter_by(
        doctor_id=current_user.id
    ).filter(
        Appointment.status.in_(['Completed', 'Rejected']),
        Appointment.appointment_date < datetime.utcnow()
    ).order_by(Appointment.appointment_date.desc()).all()
    
    active_patients_count = db.session.query(
    func.count(distinct(Appointment.patient_id))).filter(
        Appointment.doctor_id == current_user.id,
        or_(
            # Upcoming/pending appointments
            and_(
                Appointment.status.in_(['Pending', 'Approved']),
                Appointment.appointment_date >= datetime.utcnow()
            ),
            # Recently completed appointments (last 30 days)
            and_(
                Appointment.status == 'Completed',
                Appointment.appointment_date >= datetime.utcnow() - timedelta(days=30)
            )
        )
    ).scalar()

    today = date.today()
    # Get the first patient with active prescriptions (or any other logic you prefer)
    selected_patient = User.query.filter_by(role='patient').first()
    
    # Calculate active prescriptions count for this patient
    active_prescriptions_count = Prescription.query.filter_by(
        user_id=selected_patient.id, 
        is_active=True
    ).count() if selected_patient else 0
    
    return render_template('doctor_dashboard.html',
                         title='Doctor Dashboard',
                         pending_appointments=pending_appointments,
                         upcoming_appointments=upcoming_appointments,
                         past_appointments=past_appointments,
                         is_today=is_today,
                         is_soon=is_soon,
                         calculate_age=calculate_age,
                         active_patients_count = active_patients_count,
                         todays_stats= todays_stats,
                         today_date=today.strftime('%b %d, %Y'),
                         selected_patient=selected_patient,  # Add this line
                         active_prescriptions_count=active_prescriptions_count)


#########################################################################################################################################

# Admin Dashboard
#########################################################################################################################################
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('You are not authorized to view this page', 'danger')
        return redirect(url_for('index'))
    
    # Get system stats
    total_patients = User.query.filter_by(role='patient').count()
    total_doctors = User.query.filter_by(role='doctor').count()
    pending_doctors = User.query.filter_by(
        role='doctor', 
        is_approved=False
    ).count()


     # Calculate monthly appointments (last 30 days)
    monthly_appointments = Appointment.query.filter(
        Appointment.appointment_date >= datetime.now() - timedelta(days=30)
    ).count()
    
    # Get recent appointments
    recent_appointments = Appointment.query.order_by(
        Appointment.appointment_date.desc()
    ).limit(5).all()

     # Get recent activities (last 5 events)
    recent_patients = User.query.filter_by(role='patient').order_by(
        User.created_at.desc()
    ).limit(3).all()
    
    recent_appointments = Appointment.query.order_by(
        Appointment.created_at.desc()
    ).limit(3).all()
    
    # Combine and sort activities by timestamp
    activities = []
    for patient in recent_patients:
        activities.append({
            'type': 'patient_registration',
            'title': 'New patient registration',
            'description': f"{patient.first_name} {patient.surname} registered as a new patient",
            'timestamp': patient.created_at,
            'time_ago': humanize.naturaltime(datetime.now() - patient.created_at)
        })
    
    for appointment in recent_appointments:
        activities.append({
            'type': 'appointment_booked',
            'title': 'Appointment booked',
            'description': f"{appointment.patient.first_name} booked with Dr. {appointment.doctor.surname}",
            'timestamp': appointment.created_at,
            'time_ago': humanize.naturaltime(datetime.now() - appointment.created_at)
        })
    
    # Sort combined activities by time (newest first)
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    recent_activities = activities[:5]  # Get 5 most recent


    # Get pending doctors count (already in your code)
    pending_doctors = User.query.filter_by(
        role='doctor', 
        is_approved=False
    ).count()
    
    # Calculate approval progress (example: vs total doctors)
    total_doctors = User.query.filter_by(role='doctor').count()
    approval_progress = (pending_doctors / max(total_doctors, 1)) * 100

    
    return render_template('admin_dashboard.html',
                         title='Admin Dashboard',
                         total_patients=total_patients,
                         total_doctors=total_doctors,
                         pending_doctors=pending_doctors,
                         recent_appointments=recent_appointments,
                         monthly_appointments=monthly_appointments,
                         recent_activities=recent_activities,
                         approval_progress=approval_progress)

# Patient Appointments
@app.route('/patient/appointments')
@login_required
def patient_appointments():
    if current_user.role != 'patient':
        flash('Access denied', 'danger')
        return redirect(url_for('index'))
    
    page = request.args.get('page', 1, type=int)
    appointments = Appointment.query.filter_by(
        patient_id=current_user.id
    ).order_by(
        Appointment.appointment_time.desc()
    ).paginate(page=page, per_page=10)
    
    return render_template('patient_appointments.html',
                         title='My Appointments',
                         appointments=appointments)

# Doctor Appointments
@app.route('/doctor/appointments')
@login_required
def doctor_appointments():
    if current_user.role != 'doctor' or not current_user.is_approved:
        flash('Access denied', 'danger')
        return redirect(url_for('index'))
    
    status = request.args.get('status', 'scheduled')
    page = request.args.get('page', 1, type=int)
    
    appointments = Appointment.query.filter_by(
        doctor_id=current_user.id,
        status=status
    ).order_by(
        Appointment.appointment_time.asc()
    ).paginate(page=page, per_page=10)
    
    return render_template('doctor_appointments.html',
                         title='My Appointments',
                         appointments=appointments,
                         status=status)


# A route for admin to view pending doctors
@app.route('/admin/approve-doctors', methods=['GET', 'POST'])
@login_required
def approve_doctors():
    # Check if user is admin
    if current_user.role != 'admin':
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('index'))
    
    # Get all unapproved doctors
    pending_doctors = User.query.filter_by(role='doctor', is_approved=False).all()

    #get all approved doctors
    approved_doctors = User.query.filter_by(role='doctor', is_approved=True).count()
    
    #tota number of doctors
    total_doctors = User.query.filter_by(role='doctor').count()

    
    
    return render_template('admin/approve_doctors.html', 
                           pending_doctors=pending_doctors,
                           approved_doctors=approved_doctors,
                           total_doctors=total_doctors)




@app.route('/admin/approve-doctor/<int:doctor_id>', methods=['POST'])
@login_required
def approve_doctor(doctor_id):
    if current_user.role != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    doctor = User.query.get_or_404(doctor_id)
    if doctor.role != 'doctor':
        flash('User is not a doctor', 'danger')
        return redirect(url_for('approve_doctors'))
    
    doctor.is_approved = True
    db.session.commit()
    
    # Here you might want to send an approval email to the doctor
    flash(f'Doctor {doctor.first_name} {doctor.surname} has been approved', 'success')
    return redirect(url_for('approve_doctors'))


# approving/rejecting doctors
@app.route('/admin/reject-doctor/<int:doctor_id>', methods=['POST'])
@login_required
def reject_doctor(doctor_id):
    if current_user.role != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    doctor = User.query.get_or_404(doctor_id)
    if doctor.role != 'doctor':
        flash('User is not a doctor', 'danger')
        return redirect(url_for('approve_doctors'))
    
    # Here you might want to send a rejection email to the doctor
    # before deleting their account
    db.session.delete(doctor)
    db.session.commit()
    
    flash(f'Doctor {doctor.first_name} {doctor.surname} has been rejected and removed', 'warning')
    return redirect(url_for('approve_doctors'))

#Add a view doctor details route (optional)
@app.route('/admin/doctor-details/', defaults={'doctor_id': None})
@app.route('/admin/doctor-details/<int:doctor_id>')
@login_required
def view_doctor_details(doctor_id):
    if current_user.role != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    if doctor_id is None:
        # Redirect to first pending doctor or show selection
        first_pending = User.query.filter_by(role='doctor', is_approved=False).first()
        if first_pending:
            return redirect(url_for('view_doctor_details', doctor_id=first_pending.id))
        flash('No doctors available', 'info')
        return redirect(url_for('approve_doctors'))
    
    doctor = User.query.get_or_404(doctor_id)
    
    return render_template('admin/doctor_details.html', doctor=doctor)

##################################################################
# the routes for appointment booking and video consultation
##################################################################



# routes.py

from datetime import datetime
from flask import render_template, url_for, flash, redirect, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, current_user, logout_user, login_required

from partfinal.database_models import User, Appointment
from partfinal.forms import AppointmentForm, AppointmentStatusForm
from datetime import datetime, timedelta 
#######################################################################################################################################################

def send_appointment_confirmation(patient, doctor, appointment):
    """Send email confirmation for telemedicine appointments"""
    try:
        html = render_template('email/appointment_confirm.html',
                            patient=patient,
                            doctor=doctor,
                            appointment=appointment)
        text = render_template('email/appointment_confirm.txt',
                            patient=patient,
                            doctor=doctor,
                            appointment=appointment)
        
        msg = Message(
            subject=f"Telemedicine Appointment Scheduled - {appointment.appointment_date.strftime('%Y-%m-%d')}",
            recipients=[patient.email],
            html=html,
            body=text
        )
        mail.send(msg)
    except Exception as e:
        current_app.logger.error(f"Error sending appointment email: {str(e)}")

@app.route('/appointments/new', methods=['GET', 'POST'])
@login_required
def new_appointment():
    if current_user.role != 'patient':
        flash('Only patients can book appointments', 'danger')
        return redirect(url_for('index'))
    
    form = AppointmentForm()
    # Populate doctor choices with available doctors
    form.doctor_id.choices = [(doctor.id, f"Dr. {doctor.first_name} {doctor.surname} ({doctor.specialization})") 
                             for doctor in User.query.filter_by(
                                 role='doctor', 
                                 is_approved=True,
                                 
                             ).all()]
    
    # Set default duration to 30 minutes
    form.duration.data = form.duration.data or 30
    
    if form.validate_on_submit():
        try:
            # Create appointment
            appointment = Appointment(
                patient_id=current_user.id,
                doctor_id=form.doctor_id.data,
                appointment_date=form.appointment_date.data,
                duration=form.duration.data,
                symptoms=form.symptoms.data,
                notes=form.notes.data,
                status='Pending',
                created_at=datetime.utcnow()
            )
            
            db.session.add(appointment)
            db.session.commit()
            
            # Send confirmation email
            send_appointment_confirmation(
                patient=current_user,
                doctor=User.query.get(form.doctor_id.data),
                appointment=appointment
            )
            
            flash('Your telemedicine appointment has been booked! You will receive confirmation details shortly.', 'success')
            return redirect(url_for('patient_dashboard'))
            
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Error creating appointment: {str(e)}")
            flash('An error occurred while booking your appointment. Please try again.', 'danger')
    
    # Handle timezone conversion for display
    if request.method == 'GET':
        form.appointment_date.data = datetime.utcnow().replace(
            hour=9, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    
    return render_template('new_appointment.html', 
                         title='Book Telemedicine Appointment',
                         form=form,
                         timezone=app.config.get('TIMEZONE', 'UTC'))

#############################################################################################################################################

@app.route('/appointments/<int:appointment_id>', methods=['GET', 'POST'])
@login_required
def view_appointment(appointment_id):
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Check if current user is either the patient or doctor for this appointment
    if current_user.id not in [appointment.patient_id, appointment.doctor_id]:
        flash('You are not authorized to view this appointment', 'danger')
        return redirect(url_for('home'))
    
    form = None
    if current_user.role == 'doctor' and current_user.id == appointment.doctor_id:
        form = AppointmentStatusForm()
        if form.validate_on_submit():
            appointment.status = form.status.data
            if form.status.data == 'Approved':
                # Generate a unique meeting link (you might want to use a service like Zoom, Jitsi, etc.)
                appointment.meeting_link = f"https://your-telemedicine-platform.com/meeting/{appointment.id}"
            db.session.commit()
            flash('Appointment status updated!', 'success')
            return redirect(url_for('doctor_dashboard'))
    
    return render_template('view_appointment.html', 
                         title='Appointment Details',
                         appointment=appointment,
                         form=form)



@app.route('/appointments/<int:appointment_id>/cancel', methods=['POST'])
@login_required
def cancel_appointment(appointment_id):
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Verify the current user is either the patient or doctor
    if current_user.id not in [appointment.patient_id, appointment.doctor_id]:
        abort(403)
    
    # Only allow cancellation if appointment is pending or approved
    if appointment.status not in ['Pending', 'Approved']:
        flash('Cannot cancel completed or rejected appointments', 'warning')
        return redirect(url_for('view_appointment', appointment_id=appointment_id))
    
    # Update status
    appointment.status = 'Rejected'
    db.session.commit()
    
    flash('Appointment cancelled successfully', 'success')
    return redirect(url_for('patient_dashboard' if current_user.role == 'patient' else 'doctor_dashboard'))

@app.route('/appointments/<int:appointment_id>/start', methods=['GET'])
@login_required
def start_telemedicine(appointment_id):
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Check if current user is either the patient or doctor for this appointment
    if current_user.id not in [appointment.patient_id, appointment.doctor_id]:
        flash('You are not authorized to access this meeting', 'danger')
        return redirect(url_for('home'))
    
    # Check if appointment is approved and it's time for the meeting
    if appointment.status != 'Approved':
        flash('This appointment has not been approved yet', 'warning')
        return redirect(url_for('view_appointment', appointment_id=appointment.id))
    
    # Check if it's within 15 minutes before or after the scheduled time
    now = datetime.utcnow()
    if not (appointment.appointment_date - timedelta(minutes=15) <= now <= appointment.appointment_date + timedelta(minutes=appointment.duration + 15)):
        flash('The meeting can only be accessed 15 minutes before or after the scheduled time', 'warning')
        return redirect(url_for('view_appointment', appointment_id=appointment.id))
    
    return render_template('telemedicine.html', 
                         title='Telemedicine Session',
                         appointment=appointment)


# In your routes after changing appointment status
from flask import jsonify

@app.route('/appointments/<int:appointment_id>/notify', methods=['POST'])
@login_required
def notify_appointment(appointment_id):
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Check if current user is the doctor
    if current_user.id != appointment.doctor_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # In a real app, you would send email/websocket notifications here
    # For now, we'll just return a success message
    
    return jsonify({
        'success': True,
        'message': 'Notification sent to patient'
    })

#################################################################################
## Route to fetch patients 
#################################################################################
@app.route('/doctor/patients')
@login_required
def doctor_patients():
    if current_user.role != 'doctor':
        abort(403)

    patients = db.session.query(
        User,
        func.max(Appointment.appointment_date).label('last_appointment')
    ).join(
        Appointment, User.id == Appointment.patient_id
    ).filter(
        Appointment.doctor_id == current_user.id,
        Appointment.status.in_(['Approved', 'Completed'])
    ).group_by(User.id).all()

    # Convert to simple tuples
    patient_tuples = [(user, last_appt) for user, last_appt in patients]
    
    return render_template('doctor_patients.html',
                        patients=patient_tuples,
                        today=date.today())



@app.route('/patients/<int:patient_id>')
@login_required
def view_patient(patient_id):
    if current_user.role != 'doctor':
        abort(403)
    
    patient = User.query.get_or_404(patient_id)
    
    # Verify doctor-patient relationship
    has_relationship = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.patient_id == patient_id
    ).first()
    
    if not has_relationship:
        abort(403)
    
    # Get upcoming appointments
    upcoming_appointments = Appointment.query.filter(
        Appointment.doctor_id == current_user.id,
        Appointment.patient_id == patient_id,
        Appointment.status == 'Approved',
        Appointment.appointment_date >= datetime.utcnow()
    ).order_by(Appointment.appointment_date.asc()).all()
    
    return render_template('view_patient.html',
                         patient=patient,
                         upcoming_appointments=upcoming_appointments)

@app.template_filter('calculate_age')
def calculate_age(dob):
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


########################################################
# Route for submitting messages
###########################################################

# Route for doctors/admins to view messages
@app.route('/admin/messages')
@login_required
def view_messages():
    if not current_user.is_doctor and not current_user.is_admin:
        flash('You are not authorized to view this page', 'danger')
        return redirect(url_for('index'))
    
    messages = PatientMessage.query.order_by(PatientMessage.date_submitted.desc()).all()
    return render_template('admin/messages.html', messages=messages)


##############################################################################################
#route for patients to send their messages
##############################################################################################

@app.route('/message', methods=['GET', 'POST'])
def patients_message():
    form = PatientMessageForm()
    if form.validate_on_submit():
        # Create message with both form data AND automatic fields
        message = PatientMessage(
            # Form-provided data
            name=form.name.data,
            email=form.email.data,
            phone_number=form.phone_number.data,
            subject=form.subject.data,
            message=form.message.data,
            
            # System-set fields (automatically handled)
            # date_submitted will be set by default
            # is_read defaults to False
        )
        db.session.add(message)
        db.session.commit()
        flash('Message sent!', 'success')
        return redirect(url_for('predictions'))
    
    return render_template('predictions2.html', form=form)


######################################################################################
# route to fetch messages written by patients in prediction page
########################################################################################
@app.route('/doctor/messages')
@login_required
def doctor_messages():
    if not current_user.is_doctor:
        abort(403)
    
    # Get unread messages count for the card
    unread_count = PatientMessage.query.filter_by(is_read=False).count()
    
    # Get all messages sorted by date (newest first)
    messages = PatientMessage.query.order_by(PatientMessage.date_submitted.desc()).all()
    
    return render_template('doctor_dashboard.html', 
                         unread_count=unread_count,
                         messages=messages)


@app.route('/message/<int:message_id>/read')
@login_required
def mark_message_read(message_id):
    if not current_user.is_doctor:
        abort(403)
    
    message = PatientMessage.query.get_or_404(message_id)
    message.is_read = True
    message.read_by = current_user.name
    message.read_at = datetime.utcnow()
    db.session.commit()
    
    flash('Message marked as read', 'success')
    return redirect(url_for('doctor_messages'))

@app.route('/message/<int:message_id>/details')
@login_required
def message_details(message_id):
    if not current_user.is_doctor:
        abort(403)
    
    message = PatientMessage.query.get_or_404(message_id)
    return jsonify({
        'title': f"Message from {message.name}",
        'body': f"""
            <p><strong>Date:</strong> {message.date_submitted.strftime('%Y-%m-%d %H:%M')}</p>
            <p><strong>From:</strong> {message.name} &lt;{message.email}&gt;</p>
            <p><strong>Phone:</strong> {message.phone_number}</p>
            <p><strong>Subject:</strong> {message.subject}</p>
            <hr>
            <div class="message-content">
                {message.message}
            </div>
        """
    })

@app.route('/meeting')
@login_required
def meeting():
    # Get all possible parameters
    appointment_id = request.args.get('appointment_id')
    room_id = request.args.get('roomID')
    meeting_link = request.args.get('meeting_link')

    # Handle meeting_link parameter (extract room_id if needed)
    if meeting_link and not room_id:
        try:
            # Extract room_id from meeting_link (assuming format like /video_consultation/room123)
            room_id = meeting_link.split('/')[-1]
        except Exception as e:
            print(f"Error extracting room_id from meeting_link: {e}")

    # If we have an appointment_id, use that as the primary source
    if appointment_id:
        appointment = Appointment.query.get_or_404(appointment_id)
        
        # Check authorization
        if current_user.id not in [appointment.patient_id, appointment.doctor_id]:
            flash('You are not authorized to access this meeting', 'danger')
            return redirect(url_for('dashboard'))
        
        # If doctor is accessing and no room exists yet, create one
        if current_user.id == appointment.doctor_id and not appointment.video_room_id:
            appointment.start_consultation()
            db.session.commit()
        
        # Use the appointment's room_id
        room_id = appointment.video_room_id
    elif room_id:
        # Find appointment by room_id or meeting_link containing room_id
        appointment = Appointment.query.filter(
            (Appointment.video_room_id == room_id) |
            (Appointment.meeting_link.contains(room_id))
        ).first()
        
        if not appointment:
            flash('Meeting not found', 'danger')
            return redirect(url_for('join'))
            
        # Check authorization
        if current_user.id not in [appointment.patient_id, appointment.doctor_id]:
            flash('You are not authorized to join this meeting', 'danger')
            return redirect(url_for('join'))
    else:
        flash('No meeting specified', 'danger')
        return redirect(url_for('join'))

    # If we still don't have a room_id at this point, something went wrong
    if not room_id:
        flash('Could not determine meeting room', 'danger')
        return redirect(url_for('dashboard'))

    return render_template('meeting.html', roomID=room_id)

#################################################################################################################################
#################################################################################################################################
###################### join route #################################################################################
@app.route('/join', methods=['GET', 'POST'])
@login_required
def join():
    if request.method == 'POST':
        room_id = request.form.get('roomID')
        return redirect(f'/meeting?roomID={room_id}')
    
    # Get all appointments where the current user is the patient and consultation is in progress
    active_consultations = Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.status == 'InProgress',
        Appointment.video_room_id.isnot(None),
        Appointment.appointment_date >= datetime.utcnow().date()
    ).order_by(Appointment.appointment_date.desc()).all()
    
    # Prepare meeting options (both roomID and meeting_link)
    meeting_options = []
    for appointment in active_consultations:
        meeting_options.append({
            'room_id': appointment.video_room_id,
            'meeting_link': appointment.meeting_link,
            'doctor_name': appointment.doctor.surname,
            'appointment_date': appointment.appointment_date.strftime('%Y-%m-%d %H:%M')
        })
    
    return render_template('join.html',
                         meeting_options=meeting_options,
                         default_room_id=request.args.get('roomID'))
        
    #return render_template('join.html')


#################################################################################
#################################################################################
# Video Conference Routes
#################################################################################
from flask import render_template, session
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
from datetime import datetime
import hashlib
import hmac

# Initialize SocketIO
socketio = SocketIO(app)

# ZEGOCLOUD credentials
ZEGO_APP_ID = os.getenv('ZEGOCLOUD_APP_ID')
ZEGO_SERVER_SECRET = os.getenv('ZEGOCLOUD_SERVER_SECRET')

@app.route('/video_consultation/<int:appointment_id>')
@login_required
def video_consultation(appointment_id):
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Verify user has access to this appointment
    if current_user.id not in [appointment.doctor_id, appointment.patient_id]:
        abort(403)
    
    # Generate ZEGOCLOUD token
    user_id = str(current_user.id)
    room_id = str(appointment_id)
    effective_time = 3600  # 1 hour
    
    token = generate_zego_token(user_id, room_id, effective_time)
    
    return render_template('video_consultation.html', 
                         appointment=appointment,
                         zego_app_id=ZEGO_APP_ID,
                         zego_token=token,
                         room_id=room_id,
                         user_id=user_id,
                         user_name=current_user.username)

def generate_zego_token(user_id, room_id, effective_time):
    """Generate ZEGOCLOUD token for authentication"""
    timestamp = int(datetime.now().timestamp())
    signature = hmac.new(
        ZEGO_SERVER_SECRET.encode('utf-8'),
        f"{ZEGO_APP_ID}{user_id}{room_id}{timestamp}{effective_time}".encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return f"04{ZEGO_APP_ID}{user_id}{room_id}{timestamp}{effective_time}{signature}"

# SocketIO Handlers
@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('status', {'msg': f'{data["username"]} has entered the room'}, room=room)

@socketio.on('leave')
def on_leave(data):
    room = data['room']
    leave_room(room)
    emit('status', {'msg': f'{data["username"]} has left the room'}, room=room)

@socketio.on('message')
def handle_message(data):
    room = data['room']
    emit('message', {
        'username': data['username'],
        'message': data['message'],
        'timestamp': datetime.now().strftime("%H:%M")
    }, room=room)




@app.route('/save_meeting_id', methods=['POST'])
@login_required
def save_meeting_id():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        appointment_id = data.get('appointment_id')
        meeting_id = data.get('meeting_id')
        meeting_link = data.get('meeting_link')

        if not all([appointment_id, meeting_id, meeting_link]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Get the appointment
        appointment = Appointment.query.get(appointment_id)
        if not appointment:
            return jsonify({'error': 'Appointment not found'}), 404

        # Check if current user is the doctor for this appointment
        if current_user.id != appointment.doctor_id:
            return jsonify({'error': 'Unauthorized'}), 403

        # Update the appointment with meeting details
        appointment.video_room_id = meeting_id
        appointment.meeting_link = meeting_link
        appointment.status = 'InProgress'  # Update status to indicate consultation has started
        
        db.session.commit()

        return jsonify({
            'success': True,
            'message': 'Meeting details saved successfully',
            'appointment_id': appointment.id,
            'meeting_link': appointment.meeting_link,
            'video_room_id': appointment.video_room_id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    

######################################################################################################################################
#######################################################################################################################################
########################################################################################################################################

@app.route('/video_call/<room_id>')
@login_required
def video_call(room_id):
    # Find appointment by room_id
    appointment = Appointment.query.filter_by(video_room_id=room_id).first_or_404()
    
    # Verify access
    if current_user.id not in [appointment.doctor_id, appointment.patient_id]:
        abort(403)
    
    # Generate token (using your existing function)
    token = generate_zego_token(str(current_user.id), room_id, 3600)
    
    return render_template('video_call.html',
                         room_id=room_id,
                         zego_app_id=ZEGO_APP_ID,
                         zego_token=token,
                         user_id=current_user.id,
                         user_name=current_user.username,
                         appointment=appointment)

#####################################################################################################
###################### route for patient management #################################################
######################################################################################################
def calculate_age(born):
    today = datetime.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


@app.route('/admin/manage-patients')
@login_required
def manage_patients():
    if current_user.role != 'admin':
        flash('You are not authorized to view this page', 'danger')
        return redirect(url_for('index'))
    
    # Get search query from URL parameters
    search_query = request.args.get('search', '').strip()
    page = request.args.get('page', 1, type=int)
    
    # Start with base query for all patients
    query = User.query.filter_by(role='patient')
    
    # Add search filter if query exists
    if search_query:
        search_pattern = f'%{search_query}%'  # SQL LIKE pattern
        query = query.filter(
            or_(
                User.first_name.ilike(search_pattern),
                User.surname.ilike(search_pattern),
                User.email.ilike(search_pattern),
                User.phone_number.ilike(search_pattern)
            )
        )
    
    # Apply ordering and pagination
    patients = query.order_by(User.created_at.desc()).paginate(page=page, per_page=10)
    
    return render_template('admin/manage_patients.html',
                         title='Manage Patients',
                         patients=patients,
                         search_query=search_query,  # Pass back to template
                         calculate_age=calculate_age)

################################################################################################
############################# Supporting routes for patient operations:#########################
###############################################################################################


def calculate_age(born):
    today = datetime.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


@app.route('/admin/patient/<int:patient_id>',  endpoint='admin_view_patient')
@login_required
def admin_view_patient(patient_id):
    if current_user.role != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    patient = User.query.get_or_404(patient_id)
    appointments = Appointment.query.filter_by(patient_id=patient_id).order_by(Appointment.appointment_date.desc()).limit(5).all()
    
    return render_template('admin/view_patient.html',
                         patient=patient,
                         appointments=appointments,
                         calculate_age=calculate_age)

##########################################################################################################################

@app.route('/admin/patient/edit/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def edit_patient(patient_id):
    if current_user.role != 'admin':
        abort(403)  # Explicitly return 403 if not admin
    
    patient = User.query.get_or_404(patient_id)
    form = EditPatientForm(obj=patient)
    
    if form.validate_on_submit():
        try:
            form.populate_obj(patient)
            db.session.commit()
            flash('Patient updated successfully!', 'success')
            return redirect(url_for('admin_view_patient', patient_id=patient.id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating patient: {str(e)}', 'danger')
    
    return render_template('admin/edit_patient.html',
                         form=form,
                         patient=patient)

################################################################################################################

@app.route('/admin/patient/delete/<int:patient_id>', methods=['POST'])
@login_required
def delete_patient(patient_id):
    if current_user.role != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    patient = User.query.get_or_404(patient_id)
    db.session.delete(patient)
    db.session.commit()
    flash('Patient deleted successfully', 'success')
    return redirect(url_for('manage_patients'))

##########################################################################################################################################
#################################################################################################################################
#################################### ROUTE FOR RESETTING PATIENT PASSWORD ########################################################
from flask_bcrypt import Bcrypt
bcrypt = Bcrypt(app)  # Initialize bcrypt if not already done

@app.route('/admin/patient/reset-password/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def reset_patient_password(patient_id):
    if current_user.role != 'admin':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    patient = User.query.get_or_404(patient_id)
    form = ResetPasswordForm()
    
    if form.validate_on_submit():
        admin_password = request.form.get('admin_password')
        
        # Verify using bcrypt instead of Werkzeug's check_password_hash
        if not admin_password or not bcrypt.check_password_hash(current_user.password_hash, admin_password):
            flash('Admin password incorrect', 'danger')
            return redirect(url_for('reset_patient_password', patient_id=patient.id))
        
        # Update patient password using bcrypt
        patient.password_hash = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        db.session.commit()
        flash('Password updated successfully!', 'success')
        return redirect(url_for('admin_view_patient', patient_id=patient.id))
    
    return render_template('admin/reset_password.html', form=form, patient=patient)



#####################################################################################################
#####################################################################################################
######################## Route for patient profile viewing/editing: #################################
@app.route('/patient/profile', methods=['GET', 'POST'])
@login_required
def patient_profile():
    if current_user.role != 'patient':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    form = EditPatientForm(obj=current_user)
    password_form = ChangePasswordForm()
    
    # Handle profile update
    if form.validate_on_submit():
        form.populate_obj(current_user)
        db.session.commit()
        flash('Your profile has been updated!', 'success')
        return redirect(url_for('patient_profile'))
    
    # Handle password change
    if password_form.validate_on_submit():
        if bcrypt.check_password_hash(current_user.password_hash, password_form.current_password.data):
            current_user.password_hash = bcrypt.generate_password_hash(password_form.new_password.data).decode('utf-8')
            db.session.commit()
            flash('Your password has been updated!', 'success')
            return redirect(url_for('patient_profile'))
        else:
            flash('Current password is incorrect', 'danger')
    
    return render_template('patient/profile.html',
                         title='My Profile',
                         form=form,
                         password_form=password_form)


########################################################################################
########################################################################################
##################
@app.route('/patient/delete-account', methods=['POST'])
@login_required
def delete_patient_account():
    if current_user.role != 'patient':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('index'))
    
    now = datetime.utcnow()  # Define current time
    
    # Cancel all upcoming appointments
    Appointment.query.filter(
        Appointment.patient_id == current_user.id,
        Appointment.status == 'Approved',
        Appointment.created_at  > now  # Use the defined variable
    ).update({'status': 'Cancelled'})
    
    # Delete the user
    db.session.delete(current_user)
    db.session.commit()
    
    logout_user()
    flash('Your account has been permanently deleted', 'info')
    return redirect(url_for('index'))


#####################################################################################################################
##################### ROUTES FOR CHAT MESSAGES ######################################################################
#####################################################################################################################

# Add these imports at the top
from partfinal.database_models import ChatMessage, User
from partfinal.forms import ChatForm
from flask_socketio import SocketIO, emit, join_room, leave_room
import threading
import time
from sqlalchemy import inspect

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Enhanced doctor monitor with error handling
def doctor_chat_monitor():
    with app.app_context():
        while True:
            try:
                # Check if table exists first
                inspector = inspect(db.engine)
                if 'chat_messages' not in inspector.get_table_names():
                    current_app.logger.error("chat_messages table missing - waiting...")
                    time.sleep(60)
                    continue
                
                # Process emergency messages
                urgent_messages = ChatMessage.query.filter_by(
                    sender_type='user',
                    is_read=False,
                    urgency='emergency'
                ).all()
                
                for msg in urgent_messages:
                    available_doctors = User.query.filter_by(
                        role='doctor',
                        is_approved=True
                    ).all()
                    
                    if available_doctors:
                        doctor = available_doctors[0]
                        response = ChatMessage(
                            user_id=msg.user_id,
                            sender_type='doctor',
                            message=f"Dr. {doctor.first_name} is reviewing your emergency message",
                            is_read=False,
                            room_id=f"user_{msg.user_id}"
                        )
                        db.session.add(response)
                        db.session.commit()
                        
                        socketio.emit('new_message', {
                            'id': response.id,
                            'user_id': msg.user_id,
                            'message': response.message,
                            'sender': 'doctor',
                            'timestamp': response.timestamp.isoformat(),
                            'room_id': f"user_{msg.user_id}"
                        }, room=f"user_{msg.user_id}")
                
                time.sleep(10)
                
            except Exception as e:
                current_app.logger.error(f"Doctor monitor error: {str(e)}")
                db.session.rollback()
                time.sleep(30)

# Start monitor thread
if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    threading.Thread(target=doctor_chat_monitor, daemon=True).start()

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    form = ChatForm()
    room_id = f"user_{current_user.id}"
    
    if form.validate_on_submit():
        message = ChatMessage(
            user_id=current_user.id,
            sender_type='user',
            message=form.message.data,
            urgency=form.urgency.data,
            room_id=room_id
        )
        db.session.add(message)
        db.session.commit()
        
        # Broadcast message
        socketio.emit('new_message', {
            'id': message.id,
            'user_id': current_user.id,
            'message': form.message.data,
            'sender': 'user',
            'timestamp': message.timestamp.isoformat(),
            'room_id': room_id
        }, room=room_id)
        
        # AI response for non-emergencies
        if form.urgency.data != 'emergency':
            ai_response = get_ai_chat_response(form.message.data)
            ai_message = ChatMessage(
                user_id=current_user.id,
                sender_type='bot',
                message=ai_response,
                room_id=room_id
            )
            db.session.add(ai_message)
            db.session.commit()
            
            socketio.emit('new_message', {
                'id': ai_message.id,
                'user_id': current_user.id,
                'message': ai_response,
                'sender': 'bot',
                'timestamp': ai_message.timestamp.isoformat(),
                'room_id': room_id
            }, room=room_id)
        
        flash('Message sent!', 'success')
        return redirect(url_for('chat'))
    
    # Get chat history
    messages = ChatMessage.query.filter_by(room_id=room_id).order_by(ChatMessage.timestamp.asc()).all()
    
    return render_template('chat.html', form=form, messages=messages, room_id=room_id)

def get_ai_chat_response(message):
    """Enhanced AI response generator"""
    # Connect to your medical AI model here
    if any(word in message.lower() for word in ['symptom', 'pain', 'hurt']):
        return "I'm analyzing your symptoms. Could you please provide more details about: duration, severity, and any other symptoms?"
    elif any(word in message.lower() for word in ['appointment', 'schedule']):
        return "You can schedule an appointment through our booking system. Would you like me to redirect you?"
    elif any(word in message.lower() for word in ['medicine', 'prescription']):
        return "For medication questions, please provide: the name of the medication, your dosage, and any specific concerns."
    else:
        return "Thank you for your message. I'm reviewing your inquiry and will provide guidance shortly."

# SocketIO Handlers
@socketio.on('connect')
def handle_connect():
    if current_user.is_authenticated:
        room_id = f"user_{current_user.id}"
        join_room(room_id)
        emit('status', {'msg': 'Connected to chat', 'room_id': room_id})

@socketio.on('disconnect')
def handle_disconnect():
    if current_user.is_authenticated:
        leave_room(f"user_{current_user.id}")

@socketio.on('send_message')
def handle_send_message(data):
    if not current_user.is_authenticated:
        return
    
    room_id = f"user_{current_user.id}"
    message = ChatMessage(
        user_id=current_user.id,
        sender_type='user',
        message=data['message'],
        urgency=data.get('urgency', 'normal'),
        room_id=room_id
    )
    db.session.add(message)
    db.session.commit()
    
    emit('new_message', {
        'id': message.id,
        'user_id': current_user.id,
        'message': data['message'],
        'sender': 'user',
        'timestamp': message.timestamp.isoformat(),
        'room_id': room_id
    }, room=room_id)
    
    if data.get('urgency', 'normal') != 'emergency':
        ai_response = get_ai_chat_response(data['message'])
        ai_message = ChatMessage(
            user_id=current_user.id,
            sender_type='bot',
            message=ai_response,
            room_id=room_id
        )
        db.session.add(ai_message)
        db.session.commit()
        
        emit('new_message', {
            'id': ai_message.id,
            'user_id': current_user.id,
            'message': ai_response,
            'sender': 'bot',
            'timestamp': ai_message.timestamp.isoformat(),
            'room_id': room_id
        }, room=room_id)

# Doctor Chat Interface
@app.route('/doctor/chat')
@login_required
def doctor_chat():
    if current_user.role != 'doctor' or not current_user.is_approved:
        abort(403)
    
    try:
        # Get all patients with unread messages
        patients = db.session.query(User).join(ChatMessage).filter(
            ChatMessage.sender_type == 'user',
            ChatMessage.is_read == False,
            User.role == 'patient'
        ).distinct().all() or []  # Ensure this is always a list
        
        patient_id = request.args.get('patient_id')
        messages = []
        room_id = None
        current_patient = None
        
        if patient_id:
            current_patient = User.query.get(patient_id)
            if not current_patient:
                flash('Patient not found', 'danger')
                return redirect(url_for('doctor_chat'))
            
            room_id = f"user_{patient_id}"
            
            # Mark messages as read
            ChatMessage.query.filter_by(
                user_id=patient_id,
                is_read=False
            ).update({'is_read': True})
            db.session.commit()
            
            # Get conversation history - ensure this returns a list even if empty
            messages = ChatMessage.query.filter_by(
                room_id=room_id
            ).order_by(ChatMessage.timestamp.asc()).all() or []
        
        return render_template('doctor_chat.html',
                            patients=patients,
                            messages=messages,
                            current_patient=current_patient,
                            current_patient_id=int(patient_id) if patient_id else None,
                            room_id=room_id)
    
    except Exception as e:
        current_app.logger.error(f"Error in doctor_chat: {str(e)}")
        db.session.rollback()
        flash('An error occurred while loading the chat', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
# SocketIO handler for doctor joining room
@socketio.on('doctor_join')
def handle_doctor_join(data):
    if not current_user.is_authenticated or current_user.role != 'doctor':
        return
    
    room_id = data.get('room_id')
    if room_id:
        join_room(room_id)
        emit('status', {'msg': f'Doctor joined room {room_id}'}, room=room_id)

@socketio.on('doctor_send_message')
def handle_doctor_message(data):
    if not current_user.is_authenticated or current_user.role != 'doctor':
        return {'status': 'error', 'message': 'Unauthorized'}, 403
    
    try:
        # Validate message length
        if len(data['message']) > 2000:
            return {'status': 'error', 'message': 'Message too long'}, 400
            
        message = ChatMessage(
            user_id=data['patient_id'],
            sender_type='doctor',
            message=data['message'],
            room_id=data['room_id'],
            is_read=False
        )
        db.session.add(message)
        db.session.commit()
        
        emit('new_message', {
            'id': message.id,
            'message': data['message'],
            'sender': 'doctor',
            'sender_name': data.get('doctor_name', 'Doctor'),
            'timestamp': message.timestamp.isoformat(),
            'room_id': data['room_id']
        }, room=data['room_id'])
        
        return {'status': 'success'}
        
    except Exception as e:
        current_app.logger.error(f"Doctor message error: {str(e)}")
        db.session.rollback()
        return {'status': 'error', 'message': str(e)}, 500
    


########################################################################################################
########################### prescription routes ########################################################
######

# Doctor: Create new prescription
@app.route('/doctor/create_prescription/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def create_prescription(patient_id):
    if current_user.role != 'doctor':
        abort(403)
    
    patient = User.query.get_or_404(patient_id)
    form = PrescriptionForm()
    
    if form.validate_on_submit():
        prescription = Prescription(
            user_id=patient.id,
            doctor_id=current_user.id,
            medication=form.medication.data,
            dosage=form.dosage.data,
            instructions=form.instructions.data,
            prescribed_date=form.prescribed_date.data,
            is_active=form.is_active.data
        )
        db.session.add(prescription)
        db.session.commit()
        flash('Prescription created successfully!', 'success')
        return redirect(url_for('patient_prescriptions', patient_id=patient.id))
    
    return render_template('doctor/create_prescription.html', form=form, patient=patient)

# Doctor: View all prescriptions for a patient
@app.route('/doctor/patient_prescriptions/<int:patient_id>')
@login_required
def patient_prescriptions(patient_id):
    if current_user.role != 'doctor':
        abort(403)
    
    patient = User.query.get_or_404(patient_id)
    prescriptions = Prescription.query.filter_by(user_id=patient.id).order_by(Prescription.prescribed_date.desc()).all()
    return render_template('doctor/patient_prescriptions.html', patient=patient, prescriptions=prescriptions)

# Patient: View their prescriptions
@app.route('/patient/my_prescriptions')
@login_required
def my_prescriptions():
    if current_user.role != 'patient':
        abort(403)
    
    prescriptions = Prescription.query.filter_by(user_id=current_user.id).order_by(Prescription.prescribed_date.desc()).all()
    
    # Get doctor names for each prescription
    prescriptions_data = []
    for prescription in prescriptions:
        doctor = User.query.get(prescription.doctor_id)
        prescriptions_data.append({
            'prescription': prescription,
            'doctor_name': f"Dr. {doctor.surname}" if doctor else "Unknown Doctor"
        })
    
    return render_template('patient/my_prescriptions.html', 
                         prescriptions_data=prescriptions_data)

# Doctor/Patient: View single prescription
@app.route('/prescription/<int:prescription_id>')
@login_required
def view_prescription(prescription_id):
    prescription = Prescription.query.get_or_404(prescription_id)
    
    # Check if current user is either the doctor or patient for this prescription
    if current_user.id not in [prescription.user_id, prescription.doctor_id]:
        abort(403)
    
    patient = User.query.get(prescription.user_id)
    doctor = User.query.get(prescription.doctor_id)
    
    return render_template('view_prescription.html', 
                         prescription=prescription,
                         patient=patient,
                         doctor=doctor,
                         doctor_name=f"Dr. {doctor.surname}" if doctor else "Unknown Doctor")