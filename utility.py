# utility.py
def check_password(password):
    # Example: simple password check
    if password != "Secret123":
        return False
    if len(password) < 9:
        return False
    if not any(char.isdigit() for char in password):
        return False
    return True