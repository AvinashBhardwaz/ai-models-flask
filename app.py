from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import requests
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

ps = PorterStemmer()

# Load models
ipl_pipe = pickle.load(open("models\ipl_model.pkl", "rb"))  # IPL Win Predictor Model
score_pipe = pickle.load(open("models\score_pipe.pkl", "rb"))  # T20 Score Predictor Model
movies = pickle.load(open('models\movie_list.pkl', 'rb'))       # Movie 
similarity = pickle.load(open('models\similarity.pkl', 'rb'))   #Similarity Score

try:
    # Load the trained model and vectorizer
    text_vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
    spam_classifier_model = pickle.load(open('models/spam_classifier.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    text_vectorizer, spam_classifier_model = None, None

# IPL Win Predictor Data
ipl_teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

ipl_cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# T20 Score Predictor Data
t20_teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa',
    'England', 'West Indies', 'Pakistan', 'Sri Lanka'
]

t20_cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town',
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban',
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion',
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi',
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts',
    'Cardiff', 'Christchurch', 'Trinidad'
]

# Home Page - Model Selection
@app.route('/')
def home():
    return render_template('home.html')

# IPL Win Predictor Page
@app.route('/ipl_win')
def ipl_win():
    return render_template('ipl_win.html', teams=sorted(ipl_teams), cities=sorted(ipl_cities))

@app.route('/predict_ipl', methods=['POST'])
def predict_ipl():
    try:
        # Get input values safely with defaults
        batting_team = request.form.get('batting_team', '')
        bowling_team = request.form.get('bowling_team', '')
        city = request.form.get('city', '')

        target = int(request.form.get('target', '0') or 0)
        score = int(request.form.get('score', '0') or 0)
        overs = float(request.form.get('overs', '0') or 0.0)
        wickets = int(request.form.get('wickets', '0') or 0)

        # Calculate derived values
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets  # This is remaining wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Ensure correct column names match the model’s training data
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],  # Using actual lost wickets instead of wickets_left
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predict probabilities
        result = ipl_pipe.predict_proba(input_df)
        loss = result[0][0]  # Probability of losing
        win = result[0][1]   # Probability of winning

        return render_template(
            'IPL_WIN.html',
            teams=sorted(ipl_teams),
            cities=sorted(ipl_cities),
            prediction_text=f'{batting_team} - {round(win * 100)}% | {bowling_team} - {round(loss * 100)}%',
            prev_inputs={  # Retain previous inputs
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "target": target,
                "score": score,
                "overs": overs,
                "wickets": wickets  # Keeping original wickets
            }
        )

    except Exception as e:
        return jsonify({'error': str(e)})

# T20 Score Predictor Page
@app.route('/t20_score')
def t20_score():
    return render_template('t20_score.html', teams=sorted(t20_teams), cities=sorted(t20_cities))

@app.route('/predict_score', methods=['POST'])
def predict_score():
    try:
        batting_team = request.form.get('batting_team', '')
        bowling_team = request.form.get('bowling_team', '')
        city = request.form.get('city', '')

        current_score = int(request.form.get('current_score', '0') or 0)
        overs = float(request.form.get('overs', '0') or 0.0)
        wickets = int(request.form.get('wickets', '0') or 0)
        last_five = int(request.form.get('last_five', '0') or 0)

        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs if overs > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five': [last_five]
        })

        result = score_pipe.predict(input_df)
        predicted_score = int(result[0])

        return render_template('t20_score.html', 
                       teams=sorted(t20_teams), 
                       cities=sorted(t20_cities),
                       prediction_text=f'Predicted Score: {predicted_score}',
                       prev_inputs={
                           'batting_team': batting_team,
                           'bowling_team': bowling_team,
                           'city': city,
                           'current_score': current_score,
                           'overs' : overs,
                           'wickets' : wickets,
                           'balls_left': balls_left,
                           'wickets_left': wickets_left,
                           'crr': crr,
                           'last_five': last_five
                       })
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/recommend_movies', methods=['GET', 'POST'])
def recommend_movies():
    try:
        movie_list = movies['title'].values
        recommended_movies = []
        selected_movie = ''

        if request.method == 'POST':
            selected_movie = request.form.get('movie', '')  
            if selected_movie in movie_list:
                index = movies[movies['title'] == selected_movie].index[0]
                distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

                for i in distances[1:6]:
                    movie_id = movies.iloc[i[0]].movie_id
                    poster_url = fetch_poster(movie_id)
                    recommended_movies.append({"name": movies.iloc[i[0]].title, "poster": poster_url})

        return render_template(
            'movie_recommender.html', 
            movie_list=movie_list, 
            selected_movie=selected_movie, 
            recommendations=recommended_movies  # No prev_inputs needed
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def fetch_poster(movie_id):
    """Fetch movie poster from TMDB API."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=78017200c69d72f049a1d5796504694e&language=en-US"
        data = requests.get(url).json()
        #print("🔍 TMDB API Response:", data)  # Debugging API response
        poster_url = f"https://image.tmdb.org/t/p/w500/{data['poster_path']}" if 'poster_path' in data else None
        #print("🎬 Poster URL:", poster_url)  # Debugging final poster URL
        return poster_url
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching poster: {e}")
        return None
    except KeyError:
        print("⚠️ KeyError: 'poster_path' not found in response")
        return None
    
def transform_text(text):
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = [i for i in text if i.isalnum()]
        y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
        y = [ps.stem(i) for i in y]

        return " ".join(y)
    except Exception as e:
        print(f"Error in text transformation: {e}")
        return ""

@app.route('/classify_message', methods=['GET', 'POST'])
def classify_message():
    prediction = None
    input_message = ""
    
    if request.method == 'POST':
        try:
            input_message = request.form.get('message', '')
            transformed_sms = transform_text(input_message)
            vector_input = text_vectorizer.transform([transformed_sms]) if text_vectorizer else None
            result = spam_classifier_model.predict(vector_input)[0] if spam_classifier_model and vector_input is not None else None
            prediction = "Spam" if result == 1 else "Not Spam" if result == 0 else "Error in prediction"
        except Exception as e:
            print(f"Error in classification: {e}")
            prediction = "Error in processing"
    
    return render_template('email_sms_classifier.html', input_message=input_message, prediction=prediction)




if __name__ == '__main__':
    app.run(debug=True)
