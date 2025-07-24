from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from partfinal import db, login_manager, app
#from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from werkzeug.security import check_password_hash as werkzeug_check_hash
from flask_bcrypt import Bcrypt as bcrypt



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Enum for gender choices
class GenderEnum(db.Enum):
    MALE = 'male'
    FEMALE = 'female'
    OTHER = 'other'
    PREFER_NOT_TO_SAY = 'prefer_not_to_say'

# Enum for role choices
class RoleEnum(db.Enum):
    PATIENT = 'patient'
    DOCTOR = 'doctor'
    ADMIN = 'admin'


# Enum for gender choices (as class variables)
GENDER_CHOICES = ['male', 'female', 'other', 'prefer_not_to_say']

# Enum for role choices (as class variables)
ROLE_CHOICES = ['patient', 'doctor', 'admin']

# Enum for appointment status
APPOINTMENT_STATUS = ['scheduled', 'completed', 'canceled']

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(20), nullable=False)
    surname = db.Column(db.String(20), nullable=False)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.Enum(*GENDER_CHOICES, name='gender_enum'), nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    role = db.Column(db.Enum(*ROLE_CHOICES, name='role_enum'), nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships (if needed later)
    # patient_records = db.relationship('MedicalRecord', backref='patient', lazy=True)
    # doctor_appointments = db.relationship('Appointment', backref='doctor', lazy=True)
     
    #creating token for password reset

        # Doctor-specific fields
    specialization = db.Column(db.String(100), nullable=True)
    license_number = db.Column(db.String(50), nullable=True)
    hospital_affiliation = db.Column(db.String(100), nullable=True)
    is_approved = db.Column(db.Boolean, default=False)  # For admin approval
    # Admin-specific fields
    admin_code = db.Column(db.String(50), nullable=True)

    def verify_password(self, password):
        """Check password against both bcrypt and Werkzeug hashes"""
        if not self.password_hash:
            return False
        
        # Try bcrypt first
        try:
            if bcrypt.check_password_hash(self.password_hash, password):
                return True
        except:
            pass
        
        # Fall back to Werkzeug's method
        try:
            return werkzeug_check_hash(self.password_hash, password)
        except:
            return False

     # Corrected prescriptions relationship
    patient_prescriptions = db.relationship(
        'Prescription', 
        foreign_keys='Prescription.user_id',
        backref='prescribed_to', 
        lazy=True
    )
    
    doctor_prescriptions = db.relationship(
        'Prescription', 
        foreign_keys='Prescription.doctor_id',
        backref='prescribed_by', 
        lazy=True
    )

     # Relationships
    patient_appointments = db.relationship(
        'Appointment',
        foreign_keys='Appointment.patient_id',
        back_populates='patient',
        overlaps="appointments_as_patient,appointment_patient"
    )
    
    doctor_appointments = db.relationship(
        'Appointment',
        foreign_keys='Appointment.doctor_id',
        back_populates='doctor',
        overlaps="appointments_as_doctor,appointment_doctor"
    )


    def get_next_appointment(self):
        return Appointment.query.filter(
            Appointment.patient_id == self.id,
            Appointment.status == 'Approved',
            Appointment.appointment_date >= datetime.utcnow()
        ).order_by(Appointment.appointment_date.asc()).first()
   


    def get_reset_token(self, expires_sec=300):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')
    
    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']

        except:
            return None
        return User.query.get(user_id)


    def __repr__(self):
        return f"User('{self.username}', '{self.email}', '{self.role}')"
        

class Appointment(db.Model):
    __tablename__ = 'appointments'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Integer, nullable=False)  # Duration in minutes
    status = db.Column(db.Enum('Pending', 'Approved', 'Rejected', 'Completed', 'InProgress', name='appointment_status'), 
                     default='Pending')
    symptoms = db.Column(db.Text, nullable=True)
    notes = db.Column(db.Text, nullable=True)  # General appointment notes
    consultation_notes = db.Column(db.Text)  # Specific notes from video consultation
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    video_room_id = db.Column(db.String(100), nullable=True)  # Replacing meeting_link
    meeting_link = db.Column(db.String(255), nullable=True)  # Keeping for backward compatibility
    
     # Relationships
    patient = db.relationship(
        'User',
        foreign_keys=[patient_id],
        back_populates='patient_appointments',
        overlaps="appointments_as_patient,appointment_patient"
    )
    
    doctor = db.relationship(
        'User',
        foreign_keys=[doctor_id],
        back_populates='doctor_appointments',
        overlaps="appointments_as_doctor,appointment_doctor"
    )
    
    def generate_video_room_id(self):
        """Generate a unique video room ID and update meeting_link"""
        self.video_room_id = f"medcons_{self.id}_{self.doctor_id}_{self.patient_id}"
        self.meeting_link = f"/video_consultation/{self.id}"  # URL-friendly version
        db.session.commit()
        return self.video_room_id
    
    def start_consultation(self):
        """Start a video consultation by generating room details"""
        if not self.video_room_id:
            self.video_room_id = f"medcons_{self.id}_{self.doctor_id}_{self.patient_id}"
        
        if not self.meeting_link:
            self.meeting_link = f"/video_consultation/{self.id}"
        
        self.status = 'InProgress'
        db.session.commit()
        return {
            'room_id': self.video_room_id,
            'meeting_link': self.meeting_link
        }
    
    def __repr__(self):
        return f"Appointment('{self.id}', '{self.patient.username}', '{self.doctor.username}', '{self.status}')"


class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    medication = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50))
    instructions = db.Column(db.Text)
    prescribed_date = db.Column(db.Date, default=datetime.utcnow().date)
    is_active = db.Column(db.Boolean, default=True)




class PatientMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone_number = db.Column(db.String(15), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    date_submitted = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    read_by = db.Column(db.String(50))
    read_at = db.Column(db.DateTime)
    urgency = db.Column(db.String(20), default='normal')  # 'normal', 'urgent', 'emergency'

    def __repr__(self):
        return f'<PatientMessage {self.subject} from {self.name}>'
    
SENDER_TYPE_CHOICES = ['user', 'bot', 'doctor']   
class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    sender_type = db.Column(db.Enum(*SENDER_TYPE_CHOICES, name='sender_types'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)
    urgency = db.Column(db.String(20), default='normal')
    room_id = db.Column(db.String(100), nullable=True)
    
    user = db.relationship('User', backref=db.backref('chat_messages', lazy=True))
    
    def __repr__(self):
        return f"<ChatMessage {self.id} from user {self.user_id}>"