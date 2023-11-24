from flask import Flask, request, render_template
import logging
import os

app = Flask(__name__)

@app.route('/')
def home():
    os.chdir(".")
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('Team.html')

@app.route('/about')
def about():
    os.chdir(".")
    return render_template('about.html')

@app.route('/recommendations', methods=['POST'])
def runRecommendations():
    if request.method == 'POST':
        user_inputs = request.form.to_dict()
        logging.info(f"User Preference List: {user_inputs}")
        # Make sure the user_inputs have the same form as sample_input in rp/app.py
        #pivot_recommendations = runRewardandPunishmentModel(user_inputs)
        #final_recommendations = generatePreferences(pivot_recommendations)
        #graph_json_1,graph_json_2,graph_json_3= plot_recommendations(final_recommendations)

        return render_template('index.html')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)