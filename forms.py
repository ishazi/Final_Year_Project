from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField, PasswordField, TextAreaField, DateField, SelectField,  DateTimeLocalField,  DateTimeField
from wtforms.validators import DataRequired, Length, Email, EqualTo, Regexp, ValidationError, Optional
from partfinal.database_models import User
from partfinal import bcrypt
import phonenumbers
from datetime import date, datetime


# role choices
ROLE_CHOICES = [
    ('patient', 'Patient'),
    ('doctor', 'Doctor'),
    ('admin', 'Admin')
]


GENDER_CHOICES = [
    ('male', 'Male'),
    ('female', 'Female'),
    ('other', 'Other'),
    ('prefer_not_to_say', 'Prefer not to say')
]



class BaseUserForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=3, max=20)])
    surname = StringField('Surname', validators=[DataRequired(), Length(min=3, max=20)])
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    date_of_birth = DateField('Date of Birth', format='%Y-%m-%d', validators=[DataRequired()])
    gender = SelectField('Gender', choices=GENDER_CHOICES, validators=[DataRequired()])
    phone_number = StringField('Phone Number', validators=[DataRequired(), Length(min=10, max=15)])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', 
                                   validators=[DataRequired(), EqualTo('password')])
    
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')
    
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')

class PatientRegistrationForm(BaseUserForm):
    submit = SubmitField('Register as Patient')

class DoctorRegistrationForm(BaseUserForm):
    specialization = StringField('Specialization', validators=[DataRequired()])
    license_number = StringField('License Number', validators=[DataRequired()])
    hospital_affiliation = StringField('Hospital Affiliation')
    submit = SubmitField('Register as Doctor')

class AdminRegistrationForm(BaseUserForm):
    admin_code = StringField('Admin Code', validators=[DataRequired()])
    submit = SubmitField('Register as Admin')

class AppointmentForm(FlaskForm):
    doctor_id = SelectField('Doctor', coerce=int, validators=[DataRequired()])
    appointment_time = DateTimeField('Appointment Time', format='%Y-%m-%d %H:%M', 
                                   validators=[DataRequired()])
    notes = TextAreaField('Notes')
    submit = SubmitField('Book Appointment')


#Validation for the login forms
class BaseLoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Length(min=2, max=120)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Log In')

    # Common validation for all roles
    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('No account found with this email. Please register first.')
            
    def validate_password(self, password):
        user = User.query.filter_by(email=self.email.data).first()
        if user and not bcrypt.check_password_hash(user.password_hash, password.data):
            raise ValidationError('Incorrect password. Please try again.')

class PatientLoginForm(BaseLoginForm):
    # Patient-specific validation
    def validate_email(self, email):
        super().validate_email(email)  # Run base validation first
        user = User.query.filter_by(email=email.data).first()
        if user and user.role != 'patient':
            raise ValidationError('This email is not registered as a patient.')

class DoctorLoginForm(BaseLoginForm):
    license_number = StringField('License Number', validators=[Optional()])

    # Doctor-specific validation
    def validate_email(self, email):
        super().validate_email(email)  # Run base validation first
        user = User.query.filter_by(email=email.data).first()
        if user and user.role != 'doctor':
            raise ValidationError('This email is not registered as a doctor.')
            
    def validate_license_number(self, license_number):
        if license_number.data:  # Only validate if provided
            user = User.query.filter_by(email=self.email.data).first()
            if user and user.license_number != license_number.data:
                raise ValidationError('License number does not match our records.')

class AdminLoginForm(BaseLoginForm):
    admin_code = StringField('Admin Code', validators=[DataRequired()])

    # Admin-specific validation
    def validate_email(self, email):
        super().validate_email(email)  # Run base validation first
        user = User.query.filter_by(email=email.data).first()
        if user and user.role != 'admin':
            raise ValidationError('This email is not registered as an admin.')
            
    def validate_admin_code(self, admin_code):
        user = User.query.filter_by(email=self.email.data).first()
        if user and user.admin_code != admin_code.data:
            raise ValidationError('Invalid admin access code.')

class RequestResetForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')


    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is None:
            raise ValidationError('There is no account with that particular email. Please! You must register an account first.')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('ConfirmPassword', validators=[DataRequired(), Length(min=6, max=20), EqualTo('password')])
    submit = SubmitField('Reset Password')

#class AppointmentForm(FlaskForm):
   # name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    #surname = StringField('Surname', validators=[DataRequired(), Length(min=2, max=100)])
    #email = StringField('Email', validators=[DataRequired(), Email()])
    #phone = StringField('Phone Number', validators=[DataRequired(), Length(min=10, max=15)])
    
    # These would be dynamically populated in the route
    #doctor = SelectField('Select Doctor', coerce=int, validators=[DataRequired()])
    #appointment_date = DateTimeLocalField('Appointment Date', 
                                      # format='%Y-%m-%dT%H:%M',
                                      # validators=[DataRequired()])
    
    #submit = SubmitField('Book Appointment')

# forms.pyfrom flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Optional
from flask_wtf import FlaskForm
from wtforms.fields import DateTimeLocalField

from wtforms import SelectField, IntegerField, TextAreaField, SubmitField, StringField
from wtforms.validators import DataRequired, Optional, NumberRange
from flask_wtf import FlaskForm
from wtforms.fields import DateTimeLocalField
from datetime import datetime

class AppointmentForm(FlaskForm):
    doctor_id = SelectField('Doctor', 
                          coerce=int, 
                          validators=[DataRequired(message="Please select a doctor")])
    
    appointment_date = DateTimeLocalField('Appointment Date', 
                                       format='%Y-%m-%dT%H:%M', 
                                       validators=[DataRequired()],
                                       render_kw={
                                           'min': datetime.now().strftime('%Y-%m-%dT%H:%M')
                                       })
    
    duration = IntegerField('Duration (minutes)', 
                          validators=[
                              DataRequired(),
                              NumberRange(min=15, max=120, message="Duration must be 15-120 minutes")
                          ], 
                          default=30,
                          render_kw={
                              'min': 15,
                              'max': 120,
                              'step': 15
                          })
    
    symptoms = TextAreaField('Symptoms', 
                           validators=[DataRequired(message="Please describe your symptoms")],
                           render_kw={
                               'placeholder': 'Describe all symptoms in detail',
                               'rows': 5
                           })
    
    notes = TextAreaField('Additional Notes', 
                        validators=[Optional()],
                        render_kw={
                            'placeholder': 'Any special requests or information',
                            'rows': 3
                        })
    
    submit = SubmitField('Book Appointment')

    def __init__(self, *args, **kwargs):
        super(AppointmentForm, self).__init__(*args, **kwargs)
        # Dynamic doctor population
        from partfinal.database_models import User  # Import here to avoid circular imports
        self.doctor_id.choices = [
            (doc.id, f"Dr. {doc.first_name} {doc.surname} ({doc.specialization})") 
            for doc in User.query.filter_by(
                role='doctor', 
                is_approved=True
            ).order_by('surname').all()
        ]

class AppointmentStatusForm(FlaskForm):
    status = SelectField('Status', 
                       choices=[
                           ('Pending', 'Pending (Under Review)'),
                           ('Approved', 'Approve (Send Video Link)'),
                           ('Rejected', 'Reject (With Explanation)'),
                           ('InProgress', 'In Progress (Ongoing Consultation)'),
                           ('Completed', 'Complete (Finalize Notes)')
                       ], 
                       validators=[DataRequired()],
                       render_kw={
                           'class': 'form-select-lg'
                       })
    
    doctor_notes = TextAreaField('Pre-Consultation Notes', 
                               validators=[Optional()],
                               render_kw={
                                   'placeholder': 'Notes before consultation begins',
                                   'rows': 3
                               })
    
    consultation_notes = TextAreaField('Post-Consultation Notes', 
                                     validators=[Optional()],
                                     render_kw={
                                         'placeholder': 'Detailed notes from video consultation',
                                         'rows': 5
                                     })
    
    submit = SubmitField('Update Status',
                       render_kw={
                           'class': 'btn btn-primary btn-lg'
                       })





class PatientMessageForm(FlaskForm):
    name = StringField('Full Name', validators=[
        DataRequired(message="Please enter your full name"),
        Length(min=2, max=50, message="Name must be between 2-50 characters"),
        Regexp(r'^[a-zA-Z\s\-\.\']+$', 
              message="Only letters, spaces, hyphens, apostrophes and periods allowed")
    ])
    
    email = StringField('Email Address', validators=[
        DataRequired(message="Please enter your email"),
        Email(message="Please enter a valid email address"),
        Length(max=120, message="Email cannot exceed 120 characters")
    ])
    
    phone_number = StringField('Phone Number', 
    validators=[
        DataRequired(message="Please enter your phone number"),
        Length(min=10, max=15, message="Phone number must be 10-15 digits"),
        Regexp(r'^[\d\s\+\-\(\)]+$', message="Invalid phone number format")
    ],
    render_kw={
        'placeholder': 'e.g. +1 (123) 456-7890 or 0123 456 789',
        'pattern': '[\d\s\+\-\(\)]{10,15}',
        'title': 'Enter 10-15 digits with optional +, (), or spaces'
    }
)
    
    subject = StringField('Subject', validators=[
        DataRequired(message="Please enter a subject"),
        Length(min=5, max=100, message="Subject must be 5-100 characters")
    ])
    
    urgency = SelectField('Message Urgency', 
        choices=[
            ('normal', 'Normal (reply within 2 business days)'),
            ('urgent', 'Urgent (reply within 24 hours)'),
            ('emergency', 'Emergency (immediate response needed)')
        ],
        default='normal',
        validators=[DataRequired(message="Please select urgency level")]
    )
    
    message = TextAreaField('Your Message', validators=[
        DataRequired(message="Please enter your message"),
        Length(min=10, max=2000, message="Message must be 10-2000 characters")
    ])
    
    submit = SubmitField('Send Message', render_kw={"class": "btn btn-primary"})

    def validate_phone_number(self, field):
        try:
            phone = phonenumbers.parse(field.data)
            if not phonenumbers.is_valid_number(phone):
                raise ValidationError('Invalid phone number - please check the number')
        except:
            raise ValidationError('Invalid phone number format')

    def validate_message(self, field):
        if self.urgency.data == 'emergency' and len(field.data.strip()) < 30:
            raise ValidationError('For emergency messages, please provide more details (at least 30 characters)')
        if 'http://' in field.data.lower() or 'https://' in field.data.lower():
            raise ValidationError('Links are not allowed in messages for security reasons')


from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, DateField, SelectField, validators
from wtforms.validators import DataRequired, Email, Length, EqualTo

class EditPatientForm(FlaskForm):
    first_name = StringField('First Name', validators=[DataRequired(), Length(max=20)])
    surname = StringField('Surname', validators=[DataRequired(), Length(max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone_number = StringField('Phone Number', validators=[DataRequired()])
    date_of_birth = DateField('Date of Birth', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other'),  ('prefer_not_to_say', 'Prefer not to say')])

class ResetPasswordForm(FlaskForm):
    admin_password = PasswordField('Admin Password', [
        DataRequired(message="Admin password is required")
    ])
    password = PasswordField('New Password', [
        DataRequired(),
        validators.Length(min=8),
        validators.EqualTo('confirm_password', message='Passwords must match')
    ])
    confirm_password = PasswordField('Confirm Password')

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', [
        DataRequired(),
        Length(min=8),
        validators.Regexp(
            r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)',
            message='Must contain uppercase, lowercase, and numbers'
        )
    ])
    confirm_password = PasswordField('Confirm Password', [
        DataRequired(),
        EqualTo('new_password', message='Passwords must match')
    ])


class ChatForm(FlaskForm):
    message = TextAreaField('Message', validators=[
        DataRequired(),
        Length(min=1, max=1000)
    ], render_kw={
        'placeholder': 'Describe your symptoms or ask a question...',
        'rows': 3
    })
    urgency = SelectField('Urgency', choices=[
        ('normal', 'Normal (reply within 2 business days)'),
        ('urgent', 'Urgent (reply within 24 hours)'),
        ('emergency', 'Emergency (immediate response needed)')
    ], default='normal')
    submit = SubmitField('Send Message')


class PrescriptionForm(FlaskForm):
    medication = StringField('Medication', validators=[DataRequired(), Length(max=100)])
    dosage = StringField('Dosage', validators=[DataRequired(), Length(max=50)])
    instructions = TextAreaField('Instructions', validators=[DataRequired()])
    prescribed_date = DateField('Prescription Date', default=date.today)
    is_active = BooleanField('Active Prescription', default=True)