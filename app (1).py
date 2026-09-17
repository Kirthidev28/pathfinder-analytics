import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx

st.set_page_config(
    page_title="PathFinder | Career & Skill Insights",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CLEAN DARK THEME -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #07090e; color: #f1f5f9; }
    
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0d121f;
        border: 1px solid #1e293b;
        padding: 12px 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(145deg, #0e1322, #141b30);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.2;
    }
    .step-box {
        background: #0d1322;
        border-left: 4px solid #38bdf8;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        padding: 14px 18px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    .project-card {
        background: #0d1322;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD VERIFIED DATA -----------------
@st.cache_data
def load_data():
    df = pd.read_csv("pathfinder_unified_market.csv")
    df["skill_list"] = df["skills"].apply(lambda x: [s.strip() for s in str(x).split(";") if s.strip()])
    return df

df = load_data()

# Build reliable top 40 skills list
flat_skills = [s for sublist in df["skill_list"] for s in sublist]
top_skills_counts = Counter(flat_skills).most_common(40)
skill_options = sorted([s[0] for s in top_skills_counts])

# Safe defaults: ensure selected defaults actually exist in skill_options
safe_defaults = [s for s in ["Python", "Sql", "Java", "Excel"] if s in skill_options]
if not safe_defaults and len(skill_options) >= 2:
    safe_defaults = skill_options[:2]

# ----------------- SALARY REGRESSION MODEL -----------------
@st.cache_resource
def train_salary_model(data):
    sample = data.sample(min(1500, len(data)), random_state=42)
    vec = TfidfVectorizer(vocabulary=skill_options, lowercase=False)
    X_skills = vec.fit_transform(sample["skills"]).toarray()
    X_exp = sample[["experience_years"]].values
    X = np.hstack((X_exp, X_skills))
    y = sample["salary_lpa"].values
    
    rf = RandomForestRegressor(n_estimators=75, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, vec

rf_model, skill_vec = train_salary_model(df)

# ----------------- PREREQUISITE GRAPH (DAG) -----------------
PREREQ_GRAPH = nx.DiGraph([
    ("Python", "Pandas"), ("Pandas", "Numpy"), ("Numpy", "Scikit-Learn"),
    ("Scikit-Learn", "Machine Learning"), ("Machine Learning", "Deep Learning"),
    ("Sql", "Powerbi"), ("Sql", "Tableau"), ("Python", "Fastapi"),
    ("Fastapi", "Docker"), ("Docker", "Kubernetes"), ("Docker", "Aws"),
    ("Javascript", "React"), ("Javascript", "Node.Js"), ("Java", "Spring Boot")
])

def get_ordered_skills(missing):
    # Match casing
    missing_title = [s.title() for s in missing]
    sub = PREREQ_GRAPH.subgraph([s for s in missing_title if s in PREREQ_GRAPH.nodes])
    try:
        ordered = list(nx.topological_sort(sub))
    except nx.NetworkXUnfeasible:
        ordered = list(missing_title)
    remaining = [s for s in missing_title if s not in ordered]
    return ordered + remaining

# ----------------- HEADER -----------------
st.markdown(f"""
<div class="top-bar">
    <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.2rem; font-weight:800; color:#f8fafc;">🧭 PathFinder</span>
        <span style="color:#475569;">|</span>
        <span style="font-size:0.85rem; color:#94a3b8;">Career & Skill Analytics</span>
    </div>
    <div>
        <span style="font-size:0.75rem; color:#10b981; background:rgba(16,185,129,0.12); padding:4px 10px; border-radius:999px; border:1px solid rgba(16,185,129,0.25);">
            ● {len(df):,} Real Job Postings
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### **Sections**")
    menu = st.radio(
        "Choose a section to view:",
        [
            "1. Job Market Trends",
            "2. Salary Estimator",
            "3. Skill Gap & Learning Path",
            "4. Career Field Groups",
            "5. Student Readiness Check"
        ]
    )
    st.markdown("---")
    st.caption("Data Source: Verified Tech Postings (India & Global)")

# ==============================================================================
# 1. JOB MARKET TRENDS
# ==============================================================================
if menu == "1. Job Market Trends":
    st.title("Job Market Trends")
    st.caption("Key statistics on hiring demand, top skills, and salary levels.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Jobs Analyzed</div><div class="metric-val">{len(df):,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Average Salary</div><div class="metric-val">₹{df['salary_lpa'].mean():.1f} LPA</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Top Location</div><div class="metric-val">{df['location'].mode()[0]}</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">Skills Tracked</div><div class="metric-val">{len(skill_options)}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("Most In-Demand Skills")
        top_skills_df = pd.DataFrame(Counter(flat_skills).most_common(10), columns=["Skill", "Job Openings"])
        fig_bar = px.bar(
            top_skills_df, x="Job Openings", y="Skill", orientation="h",
            color="Job Openings", color_continuous_scale="Blues"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False, margin=dict(l=0, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.subheader("Salary by Experience Level")
        fig_box = px.box(
            df, x="experience_level", y="salary_lpa", points=False,
            color="experience_level", color_discrete_sequence=["#38bdf8", "#818cf8", "#c084fc"]
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1"), xaxis_title="Experience Tier", yaxis_title="Package (₹ LPA)",
            showlegend=False, margin=dict(l=0, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ==============================================================================
# 2. SALARY ESTIMATOR
# ==============================================================================
elif menu == "2. Salary Estimator":
    st.title("Salary Estimator")
    st.caption("Estimate your expected market salary based on your experience and skill set.")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Your Background")
        exp_input = st.slider("Years of Experience", 0.0, 10.0, 2.0, step=0.5)
        chosen_skills = st.multiselect("Select Skills You Know", skill_options, default=safe_defaults)

    with col2:
        st.subheader("Estimated Market Package")
        if chosen_skills:
            vec_in = skill_vec.transform([";".join(chosen_skills)]).toarray()
            X_in = np.hstack(([[exp_input]], vec_in))
            pred = rf_model.predict(X_in)[0]

            st.markdown(f"""
            <div class="metric-card" style="text-align:center; padding:24px;">
                <div class="metric-label">Estimated Market CTC</div>
                <div style="font-size:2.6rem; font-weight:800; color:#38bdf8;">₹{pred:.2f} LPA</div>
                <small style="color:#94a3b8;">Estimated Range: ₹{max(4.0, pred - 2.5):.1f} LPA – ₹{pred + 2.5:.1f} LPA</small>
            </div>
            """, unsafe_allow_html=True)

            # Top skills that add the most value
            imps = rf_model.feature_importances_
            feat_names = ["Experience"] + list(skill_vec.get_feature_names_out())
            top_i = np.argsort(imps)[-6:]
            imp_df = pd.DataFrame({
                "Skill": [feat_names[i] for i in top_i],
                "Impact Score": [imps[i] for i in top_i]
            }).sort_values("Impact Score", ascending=True)

            fig_imp = px.bar(imp_df, x="Impact Score", y="Skill", orientation="h", title="Skills Adding the Highest Value", color="Impact Score", color_continuous_scale="Teal")
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), margin=dict(l=0, r=20, t=30, b=10))
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.warning("Please pick at least one skill to see your salary estimate.")

# ==============================================================================
# 3. SKILL GAP & LEARNING PATH
# ==============================================================================
elif menu == "3. Skill Gap & Learning Path":
    st.title("Skill Gap & Learning Path")
    st.caption("See what skills you are missing for a target job and the best order to learn them.")

    top_roles = [r for r, count in Counter(df["title"]).most_common(15)]
    c1, c2 = st.columns([1, 1.2])
    with c1:
        target_role = st.selectbox("Select Your Goal Role", top_roles)
        hours_pw = st.slider("Weekly Study Hours Available", 5, 40, 15)
    with c2:
        current_skills = st.multiselect("Skills You Currently Have", skill_options, default=safe_defaults)

    if st.button("Generate Learning Plan", type="primary"):
        target_jobs = df[df["title"] == target_role]
        role_skills = [s for sublist in target_jobs["skill_list"] for s in sublist]
        market_counts = Counter(role_skills)
        target_unique = set(market_counts.keys())

        # Match score
        corpus = [" ".join(current_skills), " ".join(target_jobs["skills"].tolist())]
        tfidf = TfidfVectorizer().fit_transform(corpus)
        match_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

        missing = [s for s in target_unique if s not in current_skills]
        ordered_steps = get_ordered_skills(missing)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Profile Match</div><div class="metric-val">{match_score:.1f}%</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Skills to Learn</div><div class="metric-val">{len(missing)} Skills</div></div>""", unsafe_allow_html=True)
        with m3:
            total_h = len(missing) * 20
            wks = max(1, round(total_h / hours_pw))
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Estimated Time</div><div class="metric-val">~{wks} Weeks</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_path, col_proj = st.columns([1.1, 1])
        with col_path:
            st.subheader("Recommended Learning Order")
            st.caption("Ordered so foundational concepts come before advanced tools:")
            if not ordered_steps:
                st.success("You already have all the common skills for this role!")
            else:
                for idx, skill in enumerate(ordered_steps[:6], 1):
                    st.markdown(f"""
                    <div class="step-box">
                        <div style="display:flex; justify-content:space-between;">
                            <b>Step {idx}: {skill}</b>
                            <span style="font-size:0.75rem; color:#38bdf8;">Recommended</span>
                        </div>
                        <small style="color:#94a3b8;">Suggested effort: ~{max(1, round(20/hours_pw))} week(s)</small>
                    </div>
                    """, unsafe_allow_html=True)

        with col_proj:
            st.subheader("Recommended Practice Projects")
            st.caption("Projects that prove your skills to recruiters:")
            projects = [
                ("Interactive Analytics Dashboard", "Build an interactive web dashboard with SQL and Python to display sales and performance metrics."),
                ("Predictive ML Model Pipeline", "Create a machine learning script that cleans data, trains a model, and saves results automatically.")
            ]
            for title, desc in projects:
                st.markdown(f"""
                <div class="project-card">
                    <b style="color:#38bdf8;">📌 {title}</b>
                    <div style="font-size:0.85rem; color:#94a3b8; margin-top:4px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# 4. CAREER FIELD GROUPS
# ==============================================================================
elif menu == "4. Career Field Groups":
    st.title("Career Field Groups")
    st.caption("Visual map showing how tech job roles naturally group together based on shared skills.")

    sub_df = df.sample(min(600, len(df)), random_state=42).copy()
    vec = TfidfVectorizer(vocabulary=skill_options, lowercase=False)
    X_mat = vec.fit_transform(sub_df["skills"]).toarray()

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_mat)
    sub_df["Career Group"] = [f"Group {c+1}" for c in clusters]

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_mat)
    sub_df["PCA_1"] = coords[:, 0]
    sub_df["PCA_2"] = coords[:, 1]

    fig_cluster = px.scatter(
        sub_df, x="PCA_1", y="PCA_2", color="Career Group",
        hover_data=["title", "company"],
        color_discrete_sequence=["#38bdf8", "#818cf8", "#ec4899", "#10b981"]
    )
    fig_cluster.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"))
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("What Each Group Represents")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Group 1: Data Science & Machine Learning** (Python, Machine Learning, Deep Learning)")
        st.markdown("**Group 2: Web & Full-Stack Development** (React, Node.js, JavaScript, Java)")
    with c2:
        st.markdown("**Group 3: Business Analytics & Reporting** (SQL, PowerBI, Excel, Tableau)")
        st.markdown("**Group 4: Cloud & DevOps Infrastructure** (AWS, Docker, Linux, Kubernetes)")

# ==============================================================================
# 5. STUDENT READINESS CHECK
# ==============================================================================
elif menu == "5. Student Readiness Check":
    st.title("Student Readiness Check")
    st.caption("A quick checklist to measure your placement preparation level.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Core Foundations")
        e1 = st.checkbox("Programming Basics (Python / Java / C++)", value=True)
        e2 = st.checkbox("Writing SQL Queries (Joins, Aggregations)", value=True)
        e3 = st.checkbox("Data Cleaning & Visualization (Pandas, Excel / PowerBI)", value=True)
        e4 = st.checkbox("Basic Statistics & Math Concepts", value=False)
    with c2:
        st.subheader("Practical Skills")
        e5 = st.checkbox("Basic Docker or Linux Usage", value=False)
        e6 = st.checkbox("Building a Simple API or Web App", value=False)
        e7 = st.checkbox("Using Git and GitHub for Code", value=True)
        e8 = st.checkbox("At Least One Completed Showcase Project", value=False)

    score = int((sum([e1, e2, e3, e4, e5, e6, e7, e8]) / 8) * 100)
    st.markdown("---")
    st.subheader("Your Readiness Score")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "#64748b"},
            'bar': {'color': "#38bdf8"},
            'bgcolor': "#0d121f",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [40, 75], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ]
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc"}, height=280)
    st.plotly_chart(fig_gauge, use_container_width=True)

    if score >= 75:
        st.success("Great job! You have the core foundation needed for entry-level tech roles.")
    elif score >= 45:
        st.warning("Good progress! Building one complete project and practicing Git will get you interview-ready.")
    else:
        st.info("You're in the learning phase. Focus on programming and database basics first.")