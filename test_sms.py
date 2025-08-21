from SMS import send_sms

if __name__ == '__main__':
    phone_number = input('Enter phone number (international format, e.g. +1234567890): ')
    message = 'Test SMS from SIM7600X G-H HAT!'
    success = send_sms(phone_number, message)
    print('SMS sent successfully!' if success else 'SMS failed to send.') 