import bcrypt

hashed = bcrypt.hashpw("pass".encode(), bcrypt.gensalt())
print(hashed.decode())