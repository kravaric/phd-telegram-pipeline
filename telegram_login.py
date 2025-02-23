from telethon.sync import TelegramClient

api_id = int(input("Enter API ID: "))
api_hash = input("Enter API Hash: ")
phone_number = input("Enter Phone Number: ")

client = TelegramClient("session", api_id, api_hash)
client.connect()

if not client.is_user_authorized():
    client.send_code_request(phone_number)
    code = input("Enter the code sent to Telegram: ")
    client.sign_in(phone_number, code)

print("✅ Login successful!")
