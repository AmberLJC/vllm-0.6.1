import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
# Sender's email and password (use an "App Password" for Gmail)
sender_email = "amberjcjj@gmail.com"
sender_password = "xeja jjox blyo tbzw"
# Recipient's email address
recipient_email = "amberljc@umich.edu"

# Create a message object
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = recipient_email
message["Subject"] = "Experiment Progress"

def read_log(file='logging.txt'):
    try:
        with open(file, 'r') as file:
            lines = file.readlines()
            # Get the last ten lines or less if there are fewer lines in the file
            last_ten_lines = lines[-10:]
            return last_ten_lines
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Add the email body
job = str(sys.argv[1])
body = f"{job} is done! \n" 

# body = f"{job} is done! \n" \
#        f"Results: {read_log('/vllm/exp_logging.txt')}"
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
