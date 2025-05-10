import os
import pyotp
import imaplib
import email
from email.header import decode_header
import time

SECRET_KEY_PATH = "secret_key.txt"  # Path to store the secret key

# Email login details
EMAIL_USER = "tanjiblekamado3@gmail.com"
EMAIL_PASS = "eqwgdtqvgviptvjc"  # Ensure this is an App Password, not your normal password


def check_facebook_alerts():
    """Checks for Facebook login alert emails (Only unread emails)."""
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL_USER, EMAIL_PASS)
        mail.select("INBOX")  # Make sure you are checking the main inbox folder


        while True:  # Keep checking indefinitely
            # Search for unread emails that have "facebook" in the subject
            result, data = mail.search(None, '(UNSEEN SUBJECT "login")')


            if data and data[0]:  # If there are any unread emails
                print("🔍 Found unread emails, processing...")
                for num in data[0].split():
                    result, msg_data = mail.fetch(num, "(RFC822)")
                    raw_email = msg_data[0][1]
                    msg = email.message_from_bytes(raw_email)

                    # Decode email subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8", errors="ignore")

                    print(f"📩 Checking email with subject: {subject}")  # Debugging

                    # Only detect if subject contains a clear login alert message
                    if "login alert" in subject.lower() or "new login" in subject.lower() or "facebook" in subject.lower():
                        print("🚨 Facebook login detected!")

                        # Mark email as read after processing
                        mail.store(num, "+FLAGS", "\\Seen")

                        return True

            else:
                print("⏳ No new login alerts found. Checking again...")

            # Sleep for a while before checking again (to avoid hammering the server)
            time.sleep(5)  # Check every 10 seconds

    except imaplib.IMAP4.error as e:
        print(f"❌ IMAP Authentication Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    finally:
        if 'mail' in locals():
            try:
                mail.logout()  # Ensure the session is closed
            except Exception as e:
                print(f"❌ Error during logout: {e}")


def clean_secret_key(secret_key):
    """Cleans the secret key (removes spaces, converts to uppercase, and validates)."""
    secret_key = secret_key.replace(" ", "").upper()
    valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

    if not all(c in valid_chars for c in secret_key):
        print("❌ Invalid secret key format! Must be Base32 (A-Z, 2-7 only).")
        return None

    return secret_key


def save_secret_key(secret_key):
    """Saves the cleaned secret key to a file."""
    cleaned_key = clean_secret_key(secret_key)
    if not cleaned_key:
        return
    with open(SECRET_KEY_PATH, "w") as f:
        f.write(cleaned_key)
    print("✅ Secret key saved successfully!")


def load_secret_key():
    """Loads and cleans the secret key from the file."""
    if os.path.exists(SECRET_KEY_PATH):
        with open(SECRET_KEY_PATH, "r") as f:
            return clean_secret_key(f.read().strip())
    return None


def register_secret_key():
    """Registers a new 2FA secret key and saves it."""
    secret_key = input("Enter your Facebook 2FA Secret Key: ").strip()
    save_secret_key(secret_key)


def generate_otp():
    """Generates a TOTP token using the stored secret key."""
    secret_key = load_secret_key()
    if not secret_key:
        print("❌ No secret key found. Please register first.")
        return
    try:
        totp = pyotp.TOTP(secret_key)
        otp_code = totp.now()
        print(f"🔢 Your OTP for Facebook login: {otp_code}")
    except Exception as e:
        print(f"❌ Error generating OTP: {e}")


def monitor_facebook_login():
    """Continuously checks for login alerts and generates OTP if detected."""
    print("🔍 Monitoring Facebook login alerts... Press CTRL+C to stop.")
    try:
        while True:
            try:
                if check_facebook_alerts():
                    print("🔔 New login detected! Generating OTP...")
                    generate_otp()
                else:
                    time.sleep(10)  # If no alert, wait a bit before checking again
            except Exception as e:
                print(f"⚠️ Error in monitoring loop: {e}")
            time.sleep(60)  # Check every 60 seconds
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped.")


# 🔍 **Check for Facebook login alerts before showing the menu**
if check_facebook_alerts():
    print("🔔 New login detected! Generating OTP...")
    generate_otp()

# 🏠 **Main Menu**
while True:
    print("\n1: Register Secret Key\n2: Generate OTP\n3: Monitor Facebook Logins\n4: Exit")
    choice = input("Choose an option: ")

    if choice == "1":
        register_secret_key()
    elif choice == "2":
        generate_otp()
    elif choice == "3":
        monitor_facebook_login()  # Now runs **forever** until manually stopped
    elif choice == "4":
        break
    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")
