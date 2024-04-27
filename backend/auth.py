from flask import Flask, jsonify, request
import jwt
import psycopg2
from flask_bcrypt import Bcrypt

app = Flask(__name__)
bcrypt = Bcrypt(app)

# Global constants
global conn
app.config["SECRET_KEY"] = "jshefhbkj5456789654vgbdjgf"


conn = psycopg2.connect(
    host="localhost",
    database="snapmark",
    user="postgres",
    password='sql@2002',
    port=5432
)

def connectDB():
    global conn
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS USERS(
                id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
                email varchar(255) UNIQUE,
                name varchar(255),
                password varchar(255)
    );''')
    conn.commit()
    cur.close()

@app.route('/register', methods=["POST"])
def registration():
    data = request.json
    email = data.get('email')
    name = data.get('name')
    password = data.get('password')

    # Check if email already exists
    cur = conn.cursor()
    cur.execute('SELECT * FROM USERS WHERE email = %s', (email,))
    existing_user = cur.fetchone()
    cur.close()

    if existing_user:
        return jsonify({'message': 'Email already exists'}), 400

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Insert new user
    cur = conn.cursor()
    cur.execute( 
        '''INSERT INTO USERS
        (email, name, password) VALUES (%s, %s, %s)''', 
        (email, name, hashed_password)) 
    conn.commit()
    cur.close()

    response_data = {'message': 'Registration successful'}
    return jsonify(response_data)

@app.route('/login', methods=["POST"])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    # Check if email already exists
    cur = conn.cursor()
    cur.execute('SELECT * FROM USERS WHERE email = %s', (email,))
    existing_user = cur.fetchone()
    cur.close()

    if not existing_user:
        return jsonify({'message': 'Account not exist.'}), 400
    print(password, "   ",existing_user[3])
    get_pass=existing_user[3]
    is_valid = bcrypt.check_password_hash(get_pass,password)
    print(is_valid,'valid or not')
    if not is_valid:
        return jsonify({'message': 'Wrong Password'}), 400
    token = jwt.encode(
        {"id": existing_user[0]},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    return jsonify({'message': 'Login Successfull','token':token})
    


if __name__ == '__main__':
    connectDB()
    app.run(debug=True)
