def read_text_file():
    with open(r"D:\abhijith\ML\pravaah\client\user_chatbot_msg.txt", 'r') as file:
        content = file.read()
    return content