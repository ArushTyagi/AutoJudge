
Important - https://autojudge-sjwz8euutbxwrogida9jg5.streamlit.app/
use this link to access the portal ... It is already deployed using streamlit



🧠 AutoJudge: Predicting Programming Problem Difficulty
📌 Project Overview

Online competitive programming platforms (such as Codeforces, CodeChef, and Kattis) classify problems into difficulty levels like Easy, Medium, and Hard, often assigning a numerical difficulty score.
This process usually relies on human judgment and community feedback, which can be subjective and slow.

AutoJudge is a machine learning–based system that automatically predicts:

✅ Problem Difficulty Class (Easy / Medium / Hard)

✅ Problem Difficulty Score (numerical)

The prediction is based only on the textual description of the problem, without using submissions, tags, or user statistics.

The system also provides a simple web interface where users can paste a problem description and instantly receive predictions.

📊 Dataset Used

The dataset consists of programming problems stored in JSONL format

Each data sample contains:

title

description

input_description

output_description

problem_class (Easy / Medium / Hard)

problem_score (numerical difficulty)

🧠 Approach & Methodology

1️⃣ Data Preprocessing & Noise Handling

Combined all text fields into a single input

Cleaned text (lowercasing, symbol normalization)

Removed noisy or inconsistent samples

Handled missing values safely


2️⃣ Feature Engineering (Text-Only)

A rich hybrid feature set was designed to make the project stand out:

Text-based features

Word-level TF-IDF

Character-level TF-IDF

Handcrafted linguistic & cognitive features

Text length and lexical entropy

Keyword counts (e.g., graph, dp, recursion)

Algorithmic intent indicators

Cognitive load estimation

Constraint pressure (based on numeric limits)

Input grammar complexity

Failure-awareness signals (warnings, edge cases)


3️⃣ Classification Models (Difficulty Class)

The following models were trained and compared:

Logistic Regression

Linear SVM

RBF SVM

Random Forest Classifier

Ordinal Classifier was used with the best model

✔ Ordinal classification respects the natural ordering:

Easy < Medium < Hard

Ordinal Classification Report:

                  precision recall f1-score   support

        Easy       0.45      0.42      0.44       121
        Hard       0.58      0.79      0.67       388
      Medium        0.42      0.23      0.30       281

    accuracy                            0.53       790
   macro avg       0.48      0.48      0.47       790
weighted avg       0.51      0.53      0.50       790





4️⃣ Regression Models (Difficulty Score)

The following regressors were trained and compared:

Linear Regression

ElasticNet

Huber Regressor

Random Forest Regressor

Gradient Boosting Regressor

✔ The best regressor is selected based on MAE (Mean Absolute Error).
GradientBoostingRegressor(best model)
   MAE  = 1.6736
   RMSE = 1.9792
   R²   = 0.1237



📈 Evaluation Metrics
Classification

Accuracy

Balanced Accuracy

Precision / Recall / F1-score

Regression

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² Score

Baseline comparisons are used to ensure meaningful improvement.



🌐 Web Interface

A simple and lightweight Streamlit web application is provided.

Features:

Text boxes for:

Problem description

Input description

Output description

A Predict button

Displays:

Predicted difficulty class

Predicted difficulty score

✔ No authentication
✔ No database
✔ Uses the same trained models and preprocessing pipeline

▶️ Steps to Run the Project Locally
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/AutoJudge.git
cd AutoJudge

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

python -m training.classifier_compare

python -m training.regressor_compare


4️⃣ Run the Web Interface
streamlit run ui/app.py


The app will open automatically in your browser.

🎥 Demo Video

📽️ Demo Video (2–3 minutes):
👉 Add your demo video link here
(e.g., Google Drive / YouTube unlisted link)

👤 Author Details

Name: Arush Tyagi
Project Type: Machine Learning / Data Science
Institution: IIT Roorkee


