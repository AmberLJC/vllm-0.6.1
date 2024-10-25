import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
import subprocess

# Sender's email and password (use an "App Password" for Gmail)
sender_email = "amberjcjj@gmail.com"
sender_password = "kyuy pasf sfjr rdqj"
# Recipient's email address
recipient_email = "amberljc@umich.edu"

# Create a message object
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = recipient_email
message["Subject"] = "Experiment Progress"


def check_vllm_health():
    try:
        result = subprocess.run(['curl', 'http://localhost:8000/v1/models'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        err_str = f"An error occurred while calling curl: {e}"
        return err_str

def read_log(file='../results.log', num_lines=10):
    try:
        with open(file, 'r') as file:
            lines = file.readlines()
            # Get the last ten lines or less if there are fewer lines in the file
            last_ten_lines = lines[-num_lines:]
            return last_ten_lines
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Add the email body
job = str(sys.argv[1])


if job == 'health':
    body = f"vLLM health check: \n" \
           f"{check_vllm_health()}\n" \
           f"{read_log('results.log',5)}"
else:
    body = f"{job} is done! \n" \
        f"Results: {read_log('../results.log',20)}"
message.attach(MIMEText(body, "plain"))

try:
    # Connect to the SMTP server
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()  # Start TLS encryption
        server.login(sender_email, sender_password)  # Login to your email account

        # Send the email
        server.sendmail(sender_email, recipient_email, message.as_string())

    print("Email sent successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
