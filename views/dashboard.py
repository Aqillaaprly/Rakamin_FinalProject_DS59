import streamlit as st
import pandas as pd
import plotly.express as px

df = pd.read_csv("recruitment_efficiency_improved.csv")

BLUE = ["#1f77b4", "#4fa3d1", "#5dade2", "#2e86c1", "#2471a3", "#154360"]

def run():

    st.markdown("""
        <h1 style="text-align:center; font-size: 48px; font-weight: 800;">
            HR SUMMARY DASHBOARD
        </h1>
    """, unsafe_allow_html=True)

    # SIDEBAR METRIC
    st.markdown("### ")

    metric_style = """
        <div style="
            background-color:#e8f4fb;
            padding:20px;
            border-radius:25px;
            margin-bottom:20px;
            text-align:center;
            font-size:20px;
            font-weight:600;
            color:#1f4e79;
        ">
            {title}<br><span style="font-size:26px; font-weight:800;">{value}</span>
        </div>
    """

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(metric_style.format(
            title="Avg Cost", value=f"${df['cost_per_hire'].mean():,.0f}"
        ), unsafe_allow_html=True)

    with m2:
        st.markdown(metric_style.format(
            title="Avg Time", value=f"{df['time_to_hire_days'].mean():.0f} days"
        ), unsafe_allow_html=True)

    with m3:
        st.markdown(metric_style.format(
            title="Avg OAR", value=f"{df['offer_acceptance_rate'].mean():.2f}"
        ), unsafe_allow_html=True)

    with m4:
        st.markdown(metric_style.format(
            title="Sum Cost", value=f"${df['cost_per_hire'].sum():,.0f}"
        ), unsafe_allow_html=True)

    with m5:
        st.markdown(metric_style.format(
            title="Total Applicants", value=f"{len(df):,}"
        ), unsafe_allow_html=True)

    with m6:
        st.markdown(metric_style.format(
            title="Recruitment ID", value=f"{df['recruitment_id'].nunique()}"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # MAIN VISUALS GRAPHICS

    # FIRST ROW
    c3, c4 = st.columns(2)

    # Distribution of Department
    with c3:
        st.subheader("Distribution of Department")
        dept_count = df["department"].value_counts().reset_index()
        dept_count.columns = ["department", "count"]

        fig1 = px.pie(
            dept_count,
            names="department",
            values="count",
            color="department",
            color_discrete_sequence=BLUE,
        )
        fig1.update_traces(textposition='inside', textinfo='label+percent+value')

        st.plotly_chart(fig1, use_container_width=True)


    # OAR by Source
    with c4:
        st.subheader("OAR by Source")
        oar_df = df.groupby(["source", "department"])["offer_acceptance_rate"].mean().reset_index()

        fig2 = px.bar(
            oar_df,
            x="source",
            y="offer_acceptance_rate",
            color="department",
            barmode="group",
            color_discrete_sequence=BLUE
        )
        st.plotly_chart(fig2, use_container_width=True)

    # SECOND ROW
    c1, c2 = st.columns(2)

    # Time to Hire by Source
    with c1:
        st.subheader("Time to Hire by Source")
        time_df = df.groupby(["source", "department"])["time_to_hire_days"].mean().reset_index()

        fig3 = px.bar(
            time_df,
            x="source",
            y="time_to_hire_days",
            color="department",
            barmode="group",
            color_discrete_sequence=BLUE
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Cost per Hire by Source
    with c2:
        st.subheader("Cost per Hire by Source")
        cost_df = df.groupby(["source", "department"])["cost_per_hire"].mean().reset_index()

        fig4 = px.bar(
            cost_df,
            x="source",
            y="cost_per_hire",
            color="department",
            barmode="group",
            color_discrete_sequence=BLUE
        )
        st.plotly_chart(fig4, use_container_width=True)