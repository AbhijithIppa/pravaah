import speech_recognition as sr

recognizer= sr.Recognizer()

while True:
  try:
    with sr.Microphone() as mic:
      recognizer.adjust_for_ambient_noise(mic,duration=0.2)
      audio=recognizer.listen(mic)
      text=recognizer.recognize_google(audio)
      text=text.lower()
      with open("output.txt", "a") as file:
        file.write(text)
      print(f"{text}.")
      print('\n')
  except sr.UnknownValueError():
    recognizer=sr.Recognizer()
    continue