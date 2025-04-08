AI Model Collection - Flask Web App

This project provides an interactive web-based AI model selection system using Flask. Users can select and interact with four different AI models:

T-20 Score Predictor

IPL Win Predictor

Movie Recommender System

Email/SMS Spam Classifier

Each model is accessible through a visually appealing homepage with images for easy selection.

Project Structure

/your_project/
├── static/
│   ├── images/
│   │   ├── t20_score.jpg
│   │   ├── ipl_win.png
│   │   ├── movie_recommender.jpeg
│   │   ├── email_spam.jpg
├── templates/
│   ├── home.html
│   ├── t20_score.html
│   ├── ipl_win.html
│   ├── movie_recommender.html
│   ├── email_sms_classifier.html
├── models/
│   ├── vectorizer.pkl
│   ├── spam_classifier.pkl
│   ├── similarity.pkl
│   ├── score_pipe.pkl
│   ├── movie_list.pkl
│   ├── ipl_model.pkl
├── app.py
├── README.md
├── requirements.txt

Installation & Setup

1. Clone the Repository
git clone https://github.com/AvinashBhardwaz/ai-models-flask.git
cd ai-models-flask

2. Create Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Run the Flask App

python app.py

Models & Features

1. T-20 Score Predictor

Predicts total team score based on input features.

Uses a trained machine learning model for predictions.

Accessible via http://127.0.0.1:5000/t20_score.

2. IPL Win Predictor

Predicts the probability of a team winning an IPL match.

Uses historical data and machine learning techniques.

Accessible via http://127.0.0.1:5000/ipl_win.

3. Movie Recommender System

Recommends movies based on user input.

Uses content-based filtering and similarity scores.

Accessible via http://127.0.0.1:5000/recommend_movies.

4. Email/SMS Spam Classifier

Classifies messages as "Spam" or "Not Spam".

Uses NLP-based text processing and machine learning.

Accessible via http://127.0.0.1:5000/classify_message.

Homepage UI

The homepage (home.html) displays all models as clickable cards with images:

Each image links to its respective model page.

Navigation is seamless across models.

Users can return to the homepage anytime.

Future Enhancements

Add a database to store user interactions.

Improve UI with Bootstrap or TailwindCSS.

Deploy the application using Heroku or AWS.
