import os
import pickle
import sys
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# ---------------- PATH ---------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.predict import predict

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="AI Student Dashboard",page_icon="🎓", layout="wide")
# ---------------- ANIMATED UI ---------------- #
st.markdown("""
<h1 style='
    color:white;
    border-left:6px solid #888;
    padding-left:12px;
    font-weight:bold;
'>
Smart Student Performance AI System
</h1>
<p style='color:#aaa; margin-left:12px;'>
Advanced Analytics • Machine Learning • AI Insights
</p>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Animated Background */
.stApp {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#00c9a7);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Fade animation for header */
@keyframes fadeIn {
    from {opacity:0; transform:translateY(-20px);}
    to {opacity:1; transform:translateY(0);}
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: 0.3s;
}

/* Hover effect */
.glass:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg,#00C9A7,#92FE9D);
    color: black;
    border-radius: 10px;
    font-weight: bold;
}

/* Input boxes */
.stTextInput input {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# ---------------- CHATBOT ---------------- #
st.sidebar.title("💬 AI Assistant")

q = st.sidebar.text_input("Ask something...")

def chatbot(q):
    q = q.lower()

    # Study related
    if any(word in q for word in ["study", "hours", "time", "prepare"]):
        return "📚 Recommended study time is 6–8 hours daily with proper breaks."

    # Score related
    elif any(word in q for word in ["score", "marks", "result", "performance"]):
        return "📊 Your score depends on consistency, attendance, and previous academic performance."

    # Improvement
    elif any(word in q for word in ["improve", "improvement", "better", "increase"]):
        return "🚀 To improve performance: increase study hours, revise daily, and stay consistent."

    # Low performance
    elif any(word in q for word in ["low", "fail", "bad", "weak"]):
        return "⚠ Focus on weak subjects, revise concepts, and practice regularly."

    # High performance
    elif any(word in q for word in ["high", "top", "good", "excellent"]):
        return "🔥 Maintain consistency and keep practicing to stay ahead."

    # Sleep
    elif any(word in q for word in ["sleep", "rest"]):
        return "😴 6–8 hours of sleep is essential for better memory and focus."

    # Attendance
    elif any(word in q for word in ["attendance", "class"]):
        return "📅 Maintain attendance above 80% for better academic performance."

    # Motivation
    elif any(word in q for word in ["motivation", "lazy", "focus"]):
        return "💡 Stay disciplined, set goals, and avoid distractions."

    else:
        return "🤖 Ask me about study, score, performance, improvement, sleep or attendance."

if q:
    st.sidebar.success(chatbot(q))
    

def generate_insights(data, score):

    insights = []

    # Performance level
    if score >= 85:
        insights.append("🔥 Student is performing at an excellent level.")
    elif score >= 70:
        insights.append("👍 Student performance is good but can be improved.")
    else:
        insights.append("⚠ Student needs improvement in core subjects.")

    # Study analysis
    if data["study_hours"] < 5:
        insights.append("📚 Study hours are low compared to high performers.")

    if data["attendance"] < 75:
        insights.append("📅 Low attendance is impacting performance.")

    if data["sleep_hours"] < 6:
        insights.append("😴 Lack of sleep may reduce productivity.")

    if data["internet_usage"] > 5:
        insights.append("🌐 High internet usage may be causing distraction.")

    # Strong factor
    if data["previous_score"] > 80:
        insights.append("💡 Strong academic background detected.")

    return insights


def generate_ai_report(data, score):

    report = f"""
The student is predicted to achieve a score of {score}, which indicates a 
{"high" if score>=85 else "moderate" if score>=70 else "low"} level of performance.

The student studies for {data['study_hours']} hours daily and has an attendance of 
{data['attendance']}%. Based on this, their consistency is 
{"strong" if data['attendance']>80 else "average"}.

Their previous academic score of {data['previous_score']} suggests 
{"a solid academic background" if data['previous_score']>80 else "scope for improvement"}.

Additionally, lifestyle factors such as {data['sleep_hours']} hours of sleep and 
{data['internet_usage']} hours of internet usage indicate that 
{"they maintain a balanced routine" if data['sleep_hours']>=6 else "their routine may affect performance"}.

Overall, the student 
{"is on track for excellent results." if score>=85 else 
"can improve further with consistent effort." if score>=70 else 
"needs focused improvement strategies."}
"""

    return report

def generate_pdf(data, report_text):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    # 🔥 PROJECT HEADER
    content.append(Paragraph("Smart Student Performance AI System", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Student Performance Report", styles['Heading2']))
    content.append(Spacer(1, 20))

    # 🔥 TABLE FIRST
    table_data = [["Field", "Value"]]

    for key, value in data.items():
        table_data.append([key, str(value)])

    table = Table(table_data)

    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID",(0,0),(-1,-1),1,colors.black),
        ("BACKGROUND",(0,1),(-1,-1),colors.whitesmoke),
    ]))

    content.append(table)
    content.append(Spacer(1, 25))

    # PARAGRAPH AFTER TABLE
    content.append(Paragraph("AI Analysis Report", styles['Heading3']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(report_text, styles['Normal']))

    doc.build(content)

    return buffer
# ---------------- LOAD METRICS ---------------- #
metrics = {"MAE":0,"RMSE":0,"R2":0}
if os.path.exists("models/metrics.pkl"):
    with open("models/metrics.pkl","rb") as f:
        metrics = pickle.load(f)

# ---------------- TABS ---------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction",
    "📂 Dataset",
    "📊 Visualization",
    "🤖 Clustering",
    "📈 Model Metrics"
])

# ================= TAB 1 ================= #
with tab1:

    name = st.text_input("👤 Enter Student Name", placeholder="e.g. Daksh Vasani")

    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.slider("Study Hours",0,12,6)
        attendance = st.slider("Attendance",0,100,80)

    with col2:
        previous_score = st.slider("Previous Score",0,100,70)
        assignments = st.slider("Assignments",0,100,75)

    with col3:
        internal_marks = st.slider("Internal Marks",0,100,70)
        sleep_hours = st.slider("Sleep Hours",0,12,7)

    internet_usage = st.slider("Internet Usage",0,10,3)

    extra_activities = st.selectbox("Extra Activities",["Yes","No"])
    gender = st.selectbox("Gender",["Male","Female"])
    parent_education = st.selectbox("Parent Education",["Graduate","School"])
    family_income = st.selectbox("Family Income",["Low","Medium","High"])

    if st.button("🚀 Predict"):

        input_data = {
            "study_hours":study_hours,
            "attendance":attendance,
            "previous_score":previous_score,
            "assignments":assignments,
            "internal_marks":internal_marks,
            "sleep_hours":sleep_hours,
            "internet_usage":internet_usage,
            "extra_activities":extra_activities,
            "gender":gender,
            "parent_education":parent_education,
            "family_income":family_income
        }

        result = predict(input_data)

        st.markdown(f"""
            <div style='background:linear-gradient(90deg,#00C9A7,#92FE9D);
            padding:25px;border-radius:15px;text-align:center'>
            <h2>🎯 Predicted Score: {result}</h2>
            <p style='font-size:18px;'>Student: {name}</p>
            </div>
            """, unsafe_allow_html=True)
        st.subheader("🧠 AI Insights")
        insights = generate_insights(input_data, result)
        for i in insights:
            st.success(i)

        # -------- AI -------- #
        st.subheader("📝 AI Generated Report")

        report_text = generate_ai_report(input_data, result)

        st.markdown(f"""
            <div class="glass">
            <p style="font-size:16px; line-height:1.6">{report_text}</p>
            </div>
            """, unsafe_allow_html=True)
        # -------- PDF -------- #
        pdf = generate_pdf({
            "Name": name,
            "Study Hours": study_hours,
            "Attendance": attendance,
            "Previous Score": previous_score,
            "Assignments": assignments,
            "Internal Marks": internal_marks,
            "Sleep Hours": sleep_hours,
            "Internet Usage": internet_usage,
            "Extra Activities": extra_activities,
            "Gender": gender,
            "Parent Education": parent_education,
            "Family Income": family_income,
            "Predicted Score": result
        }, report_text)
        st.markdown(f"""""", unsafe_allow_html=True)
        st.download_button("📄 Download Report", pdf.getvalue(), "student_report.pdf")

# ================= TAB 2 ================= #
with tab2:

    raw = pd.read_csv("data/raw/student_data.csv")
    clean = pd.read_csv("data/processed/cleaned_data.csv")

    st.dataframe(raw.style.background_gradient(cmap="viridis"))
    st.download_button("Download Raw",raw.to_csv(index=False),"raw.csv")

    st.dataframe(clean.style.background_gradient(cmap="plasma"))
    st.download_button("Download Clean",clean.to_csv(index=False),"clean.csv")

# ================= TAB 3 ================= #
with tab3:

    df = pd.read_csv("data/raw/student_data.csv")
    numeric_df = df.select_dtypes(include=['int64','float64'])
    st.subheader("📊 Study Hours vs Final Score (Regression Analysis)")
    # Scatter + Regression
    x = df["study_hours"]
    y = df["final_score"]

    m,b = np.polyfit(x,y,1)

    fig = px.scatter(df,x="study_hours",y="final_score")
    fig.add_scatter(x=x,y=m*x+b,mode='lines',name='Regression')

    st.plotly_chart(fig)

    # Heatmap
    st.subheader("🔥 Feature Correlation Heatmap")
    fig2 = px.imshow(
    numeric_df.corr(),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r"
    )

    fig2.update_layout(
        height=700,
        title_x=0.5
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================= TAB 4 ================= #
with tab4:

    df = pd.read_csv("data/raw/student_data.csv")
    X = df[['study_hours','attendance','previous_score']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # KMeans
    st.subheader("🔵 KMeans Clustering Analysis")
    k = KMeans(n_clusters=3)
    df['kmeans'] = k.fit_predict(X_scaled)
    st.plotly_chart(px.scatter(df,x='study_hours',y='final_score',color='kmeans'))

    # DBSCAN
    st.subheader("🟠 DBSCAN Clustering Analysis")
    db = DBSCAN(eps=0.5)
    df['db'] = db.fit_predict(X_scaled)
    st.plotly_chart(px.scatter(df,x='study_hours',y='final_score',color='db'))

    # Hierarchical
    st.subheader("🟢 Hierarchical Clustering (Dendrogram)")
    linked = linkage(X_scaled,'ward')
    fig,ax = plt.subplots()
    dendrogram(linked)
    st.pyplot(fig)

# ================= TAB 5 ================= #
with tab5:

    col1, col2, col3 = st.columns(3)

    col1.metric("📉 MAE", round(metrics["MAE"],2))
    col2.metric("📊 RMSE", round(metrics["RMSE"],2))
    col3.metric("📈 R2 Score", round(metrics["R2"],2))

# ---------------- FOOTER ---------------- #
st.markdown("""
<hr>
<p style='text-align:left;'>
© 2026 • Developed by Daksh Vasani | Advanced Analytics | Machine Learning Enthusiast | Data Science Learner
</p>
""", unsafe_allow_html=True)