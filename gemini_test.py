import google.generativeai as genai

genai.configure(api_key="AIzaSyAcuCTZFR2Za-FqTYsnWqOUwHSLzHgdtSw")

model = genai.GenerativeModel(model_name="models/gemini-pro")
  # no 'models/' prefix
response = model.generate_content("Write a one-line poem.")
print(response.text)
