from flask import Blueprint,render_template, request, flash

auth=Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    data=request.form
    print(data)
    return render_template("login.html")

@auth.route('/register',  methods=['GET', 'POST'])
def register():
    if request.method=='POST':
        email=request.form.get('email')
        firstName=request.form.get('name')
        password=request.form.get('password')
        password2=request.form.get('password2')
        
        if len(email)<4:
            flash('email must be more than 4 characters', category='error')
            
        elif len(firstName)<2:
            flash('first name must be more than 6 characters', category='error')
        elif password != password2:
            flash('password must be same more than 4 characters', category='error')
        elif len(password)<7:
            flash('password must be more than 7 characters', category='error')
        else:
            #add user to database
            flash('Account created successfully', category='success')
            
    
    return render_template("register.html")



@auth.route('/medical-team')
def inner():
    return render_template("medical-team.html")

@auth.route('/health-info')
def healthinfo():
    return render_template("healthinfo.html")

@auth.route('/')
def home():
    return"<p>home</p>"
