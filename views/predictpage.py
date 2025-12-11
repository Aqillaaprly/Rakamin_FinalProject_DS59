import streamlit as st
import pandas as pd
import joblib
import numpy as np

df_raw = pd.read_csv("recruitment_efficiency_improved.csv") 

# LOAD MODEL
pipeline = joblib.load("models/pipeline_final_1.pkl")
model_time  = pipeline["model_time"]
model_cost  = pipeline["model_cost"]
model_oar   = pipeline["model_oar"]
scaler_cluster = pipeline["scaler_cluster"]
scaler_optimal = pipeline["scaler_optimal"]
kmeans = pipeline["kmeans"]

ohe_columns = pipeline["ohe_columns"]
preprocess_columns = pipeline["preprocess_columns"]
feature_cols_original = pipeline["feature_cols_original"]
cluster_features_columns = pipeline["cluster_features_columns"]
feature_cols_model = pipeline["feature_cols_model"]

def generate_features(dept, job, source, num_applicants, time_to_hire, cost_per_hire, oar):

    df = pd.DataFrame([{col: 0 for col in preprocess_columns}])

    df["num_applicants"] = num_applicants
    df["time_to_hire_days"] = time_to_hire
    df["cost_per_hire"] = cost_per_hire
    df["offer_acceptance_rate"] = oar

    if time_to_hire > 0:
        df["efficiency_score"] = oar / time_to_hire
    else:
        df["efficiency_score"] = 0
        
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
        
    st.markdown("<h1 style='text-align:center;font-size:45px;'>OPTIMAL SCORE PREDICTION</h1>",
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1,2.5])

    with col_left:
        
        department_jobs = {
            "Engineering": ["Software Engineer","DevOps Engineer","Backend Developer","Data Engineer"],
            "Sales": ["Account Executive","Business Development Manager","Sales Associate","Sales Representative"],
            "Product": ["Product Manager","Product Analyst","UX Designer","UI Designer"],
            "HR": ["HR Coordinator","Recruitment Specialist","Talent Acquisition","HR Manager","Payroll Specialist"],
            "Marketing": ["Marketing Specialist","Social Media Manager","Content Strategist","SEO Analyst"],
            "Finance": ["Accountant","Finance Manager","Financial Analyst","Payroll Specialist"]
        }

        st.subheader("Department")
        dept = st.selectbox("", list(department_jobs.keys()), key='dept_select')
        st.subheader("Job Title")
        job  = st.selectbox("", department_jobs[dept], key='job_select')

        st.write("----")

        time_w = st.number_input("Time Weight", 0.0, 1.0, st.session_state.time_w, step=0.01, key='time_w_input')
        cost_w = st.number_input("Cost Weight", 0.0, 1.0, st.session_state.cost_w, step=0.01, key='cost_w_input')
        oar_w  = st.number_input("OAR Weight",  0.0, 1.0, st.session_state.oar_w, step=0.01, key='oar_w_input')

        valid = abs((time_w+cost_w+oar_w)-1) < 0.001

        if valid: st.success("✔ Total weight valid")
        else:     st.error("❌ Total harus = 1")
        
        predict = st.button("PREDICT")

        def toggle_details():
            st.session_state.show_details = not st.session_state.show_details
    
    if predict and valid:
        
        data = df_raw.groupby("source").agg(
            a=("num_applicants","median"),
            t=("time_to_hire_days","median"),
            c=("cost_per_hire","median"),
            o=("offer_acceptance_rate","mean")
        ).to_dict('index')

        full = []

        for src, vals in data.items():
            X = generate_features(
                dept, job, src,
                vals["a"], vals["t"], vals["c"], vals["o"] 
            )

            pred_time = model_time.predict(X)[0]
            pred_cost = model_cost.predict(X)[0]
            pred_oar  = model_oar.predict(X)[0]

            pred_time = max(pred_time, 0.00001)
            pred_cost = max(pred_cost, 0)
            pred_oar  = min(max(pred_oar, 0), 1)

            scaled = scaler_optimal.transform([[pred_time, pred_cost, pred_oar]])
            optimal = (
                time_w * (1 - scaled[0][0]) +
                cost_w * (1 - scaled[0][1]) +
                oar_w * scaled[0][2]
            )

            full.append([
                src, 
                round(pred_time), 
                round(pred_cost), 
                round(pred_oar,2), 
                round(optimal,4),
                scaled[0][0], scaled[0][1], scaled[0][2]
            ])

        df = pd.DataFrame(full, columns=[
            "Source","Pred Time","Pred Cost","Pred OAR","Optimal Score",
            "Scaled Time", "Scaled Cost", "Scaled OAR"
        ])
        df = df.sort_values(by="Optimal Score", ascending=False)
        
        st.session_state.prediction_run = True
        st.session_state.df_result = df
        st.session_state.best_result = df.iloc[0]
        st.session_state.dept = dept
        st.session_state.job = job
        st.session_state.time_w = time_w
        st.session_state.cost_w = cost_w
        st.session_state.oar_w = oar_w
        st.session_state.show_details = False 
    
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
            
            # CUSTOM TABLE CSS
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
            .custom-table tbody tr:nth-child(even) {
                background-color: #f5f7ff;
            }
            .custom-table tbody tr:nth-child(odd) {
                background-color: #ffffff;
            }
            .custom-table tbody tr:hover {
                background-color: #e9edff;
                cursor: pointer;
            }
            </style>
            """
            st.markdown(table_css, unsafe_allow_html=True)
            
            df_display = df[["Source","Pred Time","Pred Cost","Pred OAR","Optimal Score"]]
            html_table = df_display.to_html(index=False, classes="custom-table")
            st.markdown(html_table, unsafe_allow_html=True)

            st.write("---")
            
            
            # CSS rec card
            st.markdown(f"""
                <style>
                .recommendation-card-new {{
                    background-color: #f0f8ff; 
                    padding: 20px;
                    border-radius: 12px;
                    border-left: 8px solid #4682B4; 
                    margin-bottom: 5px; /* Kurangi margin bawah agar dekat tombol */
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
                    /* Tambahkan warna teks default untuk card */
                    color: #2C3E50; 
                }}
                .best-source-text-new {{
                    font-size: 32px;
                    font-weight: 800;
                    color: #2C3E50; /* Warna teks gelap */
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

            st.button("Lihat Penjelasan", on_click=toggle_details, use_container_width=True, key='toggle_button') # <-- Ganti key-nya

            st.markdown("### Detail Prediksi")
            c_met1, c_met2, c_met3, c_met4 = st.columns(4)

            def metric_card_new(title, value, unit, color="#1f4e79"):
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

            with c_met1:
                st.markdown(metric_card_new("Time", f"{time_val:.0f}", "days"), unsafe_allow_html=True)
            with c_met2:
                st.markdown(metric_card_new("Cost", f"${cost_val:.0f}", ""), unsafe_allow_html=True)
            with c_met3:
                st.markdown(metric_card_new("OAR", f"{oar_val:.2f}", ""), unsafe_allow_html=True)
            with c_met4:
                st.markdown(metric_card_new("Optimal Score", f"{score_val:.4f}", "", color="#b00000"), unsafe_allow_html=True)
                
            st.write("----")

            # PENJELASAN DETAIL
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
                    - **Predicted Time (Waktu):** {time_val:.0f} hari (dengan bobot {time_w_used*100:.0f}%)
                    - **Predicted Cost (Biaya):** ${cost_val:.0f} (dengan bobot {cost_w_used*100:.0f}%)
                    - **Predicted OAR (Tingkat Penerimaan):** {oar_val:.2f} (dengan bobot {oar_w_used*100:.0f}%)
                """)
                
                st.markdown(f"**Optimal Score** dihitung sebagai:")
                st.latex(f"""
                    \\text{{Score}} = ({time_w_used} \\times (1 - {t_scaled:.2f})) + ({cost_w_used} \\times (1 - {c_scaled:.2f})) + ({oar_w_used} \\times {o_scaled:.2f}) \\approx {score_val:.4f}
                """)