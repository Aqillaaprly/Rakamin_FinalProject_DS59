import streamlit as st
import pandas as pd
import joblib
import numpy as np


#  LOAD PIPELINE
pipeline = joblib.load("models/pipeline_final_1.pkl")
model_time  = pipeline["model_time"]
model_cost  = pipeline["model_cost"]
model_oar   = pipeline["model_oar"]
scaler_cluster = pipeline["scaler_cluster"]
scaler_optimal = pipeline["scaler_optimal"]
kmeans = pipeline["kmeans"]

preprocess_columns = pipeline["preprocess_columns"]
cluster_features_columns = pipeline["cluster_features_columns"]
feature_cols_model = pipeline["feature_cols_model"]

# FEATURE GENERATOR
def generate_features(dept, job, source, num_applicants, time_to_hire, cost_per_hire, oar):
    df = pd.DataFrame([{col: 0 for col in preprocess_columns}])
    df["num_applicants"] = num_applicants
    df["time_to_hire_days"] = time_to_hire
    df["cost_per_hire"] = cost_per_hire
    df["offer_acceptance_rate"] = oar
    df["efficiency_score"] = oar / max(time_to_hire, 0.0001)
    df["time_cost_interaction"] = cost_per_hire * time_to_hire

    dcol = f"department_{dept}"
    jcol = f"job_title_{job}"
    scol = f"source_{source}"

    if dcol in df.columns: df[dcol] = 1
    if jcol in df.columns: df[jcol] = 1
    if scol in df.columns: df[scol] = 1

    df_cluster_input = df[cluster_features_columns]
    X_scaled = scaler_cluster.transform(df_cluster_input)
    df["cluster"] = kmeans.predict(X_scaled)

    df_final = df.reindex(columns=feature_cols_model, fill_value=0)
    return df_final

#  RECOMMENDATION TEXT 
def get_recommendation_text(source):
    if source == "Referral":
        return "Referral menjadi pilihan paling optimal karena ia memberikan kualitas kandidat yang lebih terjamin. Kandidat dari referral umumnya sudah memiliki konteks budaya kerja dan rekomendasi internal, sehingga proses adaptasi lebih cepat dan tingkat penerimaan tawaran jauh lebih tinggi. Dengan kombinasi biaya yang rendah, kualitas kandidat yang solid, serta efisiensi proses, Referral layak diprioritaskan sebagai kanal utama dalam strategi rekrutmen kita."
    elif source == "Job Portal":
        return "Job Portal menjadi kanal paling optimal karena menawarkan proses hiring tercepat dengan biaya yang tetap efisien. Kandidat dari Job Portal juga menunjukkan tingkat penerimaan tawaran tertinggi, sehingga meminimalkan risiko pengulangan proses rekrutmen. Dengan perpaduan kecepatan, efisiensi, dan kepastian hiring, Job Portal sangat direkomendasikan sebagai kanal utama untuk mempercepat pemenuhan kebutuhan talent."
    elif source == "LinkedIn":
        return "LinkedIn menempati posisi teratas karena mampu menghadirkan kandidat dengan kualitas profesional yang lebih tinggi. Meskipun biayanya relatif lebih besar, kualitas dan relevansi kandidat yang lebih kuat membuat proses seleksi lebih akurat dan efektif. Dengan tingkat kecocokan yang tinggi terhadap kompetensi teknis dan budaya perusahaan, LinkedIn sangat direkomendasikan ketika prioritas utama adalah menemukan kandidat berkualitas premium."
    elif source == "Recruiter":
        return "Rekruter menjadi kanal optimal karena mampu menyediakan kandidat yang sudah tersaring secara profesional. Mereka biasanya mengirim talenta yang lebih siap dan relevan, sehingga mengurangi beban tim internal dalam screening awal. Meski biaya sedikit lebih tinggi, efisiensi proses dan kualitas kandidat menjadikan Recruiter sebagai pilihan terbaik ketika kebutuhan hiring bersifat mendesak atau memerlukan kualifikasi teknis yang spesifik."
    else:
        return "Penjelasan kontekstual tidak tersedia untuk sumber ini."

def run():
    if 'prediction_run' not in st.session_state:
        st.session_state.prediction_run = False
        st.session_state.df_result = None
        st.session_state.best_result = None
        st.session_state.dept = None
        st.session_state.job = None
        st.session_state.time_w = 0.4
        st.session_state.cost_w = 0.4
        st.session_state.oar_w = 0.2
        st.session_state.show_details = False
        
    st.markdown("<h1 style='text-align:center;font-size:45px;'>OPTIMAL SCORE PREDICTION</h1>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1,2.5])

    with col_left:
        st.subheader("Upload Dataset (.csv)")
        uploaded = st.file_uploader("", type=["csv"])

        source_data = None

        if uploaded:
            df_user = pd.read_csv(uploaded)
            required = ["time_to_hire_days","cost_per_hire","offer_acceptance_rate", "source"]
            
            if not all(col in df_user.columns for col in required):
                st.error(f"❌ Dataset harus memiliki kolom: {', '.join(required)}")
                return

            st.success("✔ Dataset berhasil dibaca")
            st.subheader("Uploaded Dataset Preview")
            st.dataframe(df_user.head(), use_container_width=True)

            unique_sources = df_user["source"].unique()
            
            if len(unique_sources) < 4:
                st.warning(f"⚠️ Dataset hanya memiliki {len(unique_sources)} source: {list(unique_sources)}")
                st.info("💡 Untuk hasil optimal, dataset sebaiknya mencakup: Referral, LinkedIn, Job Portal, Recruiter")

            source_data = {}
            for src in unique_sources:
                df_src = df_user[df_user["source"] == src]
                source_data[src] = {
                    "a": len(df_src),
                    "t": df_src["time_to_hire_days"].median(),
                    "c": df_src["cost_per_hire"].median(),
                    "o": df_src["offer_acceptance_rate"].mean()
                }
            
        department_jobs = {
            "Engineering": ["Software Engineer","DevOps Engineer","Backend Developer","Data Engineer"],
            "Sales": ["Account Executive","Business Development Manager","Sales Associate","Sales Representative"],
            "Product": ["Product Manager","Product Analyst","UX Designer","UI Designer"],
            "HR": ["HR Coordinator","Recruitment Specialist","Talent Acquisition","HR Manager","Payroll Specialist"],
            "Marketing": ["Marketing Specialist","Social Media Manager","Content Strategist","SEO Analyst"],
            "Finance": ["Accountant","Finance Manager","Financial Analyst","Payroll Specialist"]
        }

        st.subheader("Department")
        dept = st.selectbox("", list(department_jobs.keys()), key="dept_up")
        st.subheader("Job Title")
        job  = st.selectbox("", department_jobs[dept], key="job_up")

        st.write("----")

        time_w = st.number_input("Time Weight", 0.0, 1.0, st.session_state.time_w, step=0.01, key="w_time_up")
        cost_w = st.number_input("Cost Weight", 0.0, 1.0, st.session_state.cost_w, step=0.01, key="w_cost_up")
        oar_w  = st.number_input("OAR Weight", 0.0, 1.0, st.session_state.oar_w, step=0.01, key="w_oar_up")

        valid = abs((time_w+cost_w+oar_w)-1) < 0.001
        if valid: st.success("✔ Total weight valid")
        else: st.error("❌ Total harus = 1")

        predict = st.button("PREDICT", disabled=not(uploaded and valid), key="predict_up")

    def toggle_details():
        st.session_state.show_details = not st.session_state.show_details

    # --- PREDICTION LOGIC ---
    if predict and uploaded and valid:
        full = []
        all_pred_time = []
        all_pred_cost = []
        all_pred_oar = []

        for src, vals in source_data.items():
            X = generate_features(dept, job, src, vals["a"], vals["t"], vals["c"], vals["o"])

            pred_time_raw = model_time.predict(X)[0]
            pred_cost_raw = model_cost.predict(X)[0]
            pred_oar_raw = model_oar.predict(X)[0]
            
            pred_time = max(pred_time_raw, 0.00001)
            pred_cost = max(pred_cost_raw, 0)
            pred_oar  = min(max(pred_oar_raw, 0), 1)
            
            all_pred_time.append(pred_time)
            all_pred_cost.append(pred_cost)
            all_pred_oar.append(pred_oar)
        
        time_min, time_max = min(all_pred_time), max(all_pred_time)
        cost_min, cost_max = min(all_pred_cost), max(all_pred_cost)
        oar_min, oar_max = min(all_pred_oar), max(all_pred_oar)
        
        time_range = max(time_max - time_min, 0.00001)
        cost_range = max(cost_max - cost_min, 0.00001)
        oar_range = max(oar_max - oar_min, 0.00001)
        
        for i, (src, vals) in enumerate(source_data.items()):
            pred_time = all_pred_time[i]
            pred_cost = all_pred_cost[i]
            pred_oar = all_pred_oar[i]
            
            time_scaled = (pred_time - time_min) / time_range
            cost_scaled = (pred_cost - cost_min) / cost_range
            oar_scaled = (pred_oar - oar_min) / oar_range
            
            optimal = (
                time_w * (1 - time_scaled) +
                cost_w * (1 - cost_scaled) +
                oar_w * oar_scaled
            )

            full.append([
                src, 
                round(pred_time), 
                round(pred_cost), 
                round(pred_oar, 2), 
                round(optimal, 4),
                time_scaled, cost_scaled, oar_scaled
            ])

        df_pred = pd.DataFrame(full, columns=[
            "Source","Pred Time","Pred Cost","Pred OAR","Optimal Score",
            "Scaled Time", "Scaled Cost", "Scaled OAR"
        ])
        df_pred = df_pred.sort_values(by="Optimal Score", ascending=False)
        
        st.session_state.prediction_run = True
        st.session_state.df_result = df_pred
        st.session_state.best_result = df_pred.iloc[0]
        st.session_state.dept = dept
        st.session_state.job = job
        st.session_state.time_w = time_w
        st.session_state.cost_w = cost_w
        st.session_state.oar_w = oar_w
        st.session_state.show_details = False

    # RIGHT COLUMN
    with col_right:
        st.subheader("Prediction Result")
        
        if st.session_state.prediction_run:
            df = st.session_state.df_result
            best = st.session_state.best_result
            
            best_source = best["Source"]
            score_val = round(best["Optimal Score"], 4)
            time_val = round(best["Pred Time"], 2)
            cost_val = round(best["Pred Cost"], 2)
            oar_val = round(best["Pred OAR"], 2)
            
            time_w_used = st.session_state.time_w
            cost_w_used = st.session_state.cost_w
            oar_w_used = st.session_state.oar_w
            
            dept_used = st.session_state.dept
            job_used = st.session_state.job
            
            # CSS TABLE
            table_css = """
            <style>
            .custom-table {
                width: 100%;
                border-collapse: collapse;
                border-radius: 14px;
                overflow: hidden;
                font-family: 'Arial';
                margin-top: 10px;
            }
            .custom-table thead {
                background-color: #6C8CFF;
                color: #000000 !important;
                font-weight: bold;
                text-align: left;
            }
            .custom-table th, .custom-table td {
                padding: 14px 20px;
                font-size: 16px;
                color: #000000 !important;
                text-align: center;
            }
            .custom-table tbody tr:nth-child(even) { background-color: #f5f7ff; }
            .custom-table tbody tr:nth-child(odd) { background-color: #ffffff; }
            .custom-table tbody tr:hover { background-color: #e9edff; cursor: pointer; }
            </style>
            """
            st.markdown(table_css, unsafe_allow_html=True)
            
            df_display = df[["Source","Pred Time","Pred Cost","Pred OAR","Optimal Score"]]
            st.markdown(df_display.to_html(index=False, classes="custom-table"), unsafe_allow_html=True)

            st.write("---")
            
            # RECOMMENDATION CARD
            st.markdown(f"""
                <style>
                .recommendation-card-new {{
                    background-color: #f0f8ff; 
                    padding: 20px;
                    border-radius: 12px;
                    border-left: 8px solid #4682B4; 
                    margin-bottom: 5px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    color: #2C3E50; 
                }}
                .best-source-text-new {{
                    font-size: 32px;
                    font-weight: 800;
                    color: #2C3E50;
                    margin-top: 5px;
                    margin-bottom: 5px;
                }}
                </style>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="recommendation-card-new">
                    <p style="font-size: 30px; margin: 0; color: #1D1D86; font-weight: 600;">Best Source Recommendation for</p>
                    <p style="font-size: 24px; font-weight: 700; margin: 0;">{job_used} in {dept_used} Department is</p>
                    <p style="font-size: 26px; font-weight: 700; margin: 0;color: #b00000">{best_source}</p>
                </div>
            """, unsafe_allow_html=True)

            st.button("Lihat Penjelasan", on_click=toggle_details, use_container_width=True, key='toggle_up')

            st.markdown("### Detail Prediksi")
            c1, c2, c3, c4 = st.columns(4)

            def metric_card(title, value, unit, color="#1f4e79"):
                return f"""
                    <div style="
                        background-color:#e8f4fb;
                        padding:15px 5px;
                        border-radius:10px;
                        text-align:center;
                        font-size:14px;
                        font-weight:600;
                        color:{color};
                        height: 100px;
                    ">
                        {title}<br>
                        <span style="font-size:20px; font-weight:800;">{value}</span><br>
                        <span style="font-size:12px;">{unit}</span>
                    </div>
                """

            with c1:
                st.markdown(metric_card("Time", f"{time_val:.0f}", "days"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card("Cost", f"${cost_val:.0f}", ""), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("OAR", f"{oar_val:.2f}", ""), unsafe_allow_html=True)
            with c4:
                st.markdown(metric_card("Optimal Score", f"{score_val:.4f}", "", color="#b00000"), unsafe_allow_html=True)
                
            st.write("----")

            # DETAILED EXPLANATION
            if st.session_state.show_details:
                t_scaled = best["Scaled Time"]
                c_scaled = best["Scaled Cost"]
                o_scaled = best["Scaled OAR"]
                
                st.markdown("### 📋 Detail Rekomendasi")
                
                st.info(f"Rekomendasi: **{best_source}** (Optimal Score: **{score_val:.4f}**)")
                st.markdown(f"Untuk Jabatan **{job_used}** di **{dept_used}**, **{best_source}** memberikan keseimbangan terbaik dari metrik yang diprediksi.")
                
                explanation = get_recommendation_text(best_source)
                st.markdown(f"**Alasan Prioritas:** {explanation}")
                st.write("---") 
                
                st.markdown(f"""
                    - **Predicted Time (Waktu):** {time_val:.0f} hari ({round(100*(1-t_scaled), 1)}% skor, bobot {time_w_used*100:.0f}%)
                    - **Predicted Cost (Biaya):** ${cost_val:.0f} ({round(100*(1-c_scaled), 1)}% skor, bobot {cost_w_used*100:.0f}%)
                    - **Predicted OAR (Tingkat Penerimaan):** {oar_val:.2f} ({round(100*o_scaled, 1)}% skor, bobot {oar_w_used*100:.0f}%)
                """)
                
                st.markdown(f"**Optimal Score** dihitung sebagai:")
                st.latex(f"""
                    \\text{{Score}} = ({time_w_used} \\times (1 - {t_scaled:.2f})) + ({cost_w_used} \\times (1 - {c_scaled:.2f})) + ({oar_w_used} \\times {o_scaled:.2f}) \\approx {score_val:.4f}
                """)
        else:
            st.info("👆 Upload dataset dan klik PREDICT untuk melihat hasil")