"""
ForeWatt Dashboard - Unified Data Explorer Page
Explore historical data, features, and patterns for both Consumption and Price.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    load_master_data, get_data_summary, get_feature_groups,
    apply_date_filter, get_date_range_options,
    TARGET_VARIABLE, PAGE_CONFIG, create_time_series_plot,
    create_correlation_heatmap, create_hourly_pattern_plot, create_box_plot,
    PROFESSIONAL_CSS, HIDE_OPTIMIZATION_NAV_CSS, get_model_colors,
    render_page_header, render_divider
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Apply professional styling
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)

# Check session state for model selection
if 'selected_model' not in st.session_state or st.session_state.selected_model is None:
    st.warning("Please select a model type first.")
    if st.button("Go to Home", use_container_width=True):
        st.switch_page("Home.py")
    st.stop()

# Get model type from session state
MODEL_TYPE = st.session_state.selected_model

# Hide Optimization Architecture from sidebar for consumption mode
if MODEL_TYPE == 'consumption':
    st.markdown(HIDE_OPTIMIZATION_NAV_CSS, unsafe_allow_html=True)

# Configure based on model type
colors = get_model_colors(MODEL_TYPE)
if MODEL_TYPE == 'consumption':
    TARGET = "consumption"
    ICON = "⚡"
    TITLE = "Consumption Data Explorer"
    COLOR = colors['primary']
    UNIT = "MWh"
    TARGET_LABEL = "Electricity Consumption"
else:
    TARGET = "price_real"
    ICON = "💰"
    TITLE = "Price Data Explorer"
    COLOR = colors['primary']
    UNIT = "TL/MWh"
    TARGET_LABEL = "Electricity Price (PTF)"

# Header with back button
col1, col2 = st.columns([6, 1])
with col1:
    render_page_header(ICON, TITLE, "Explore historical data, patterns, and feature analysis", COLOR)
with col2:
    if st.button("← Back", key="back_menu_data", use_container_width=True):
        st.switch_page("Home.py")

try:
    # Load data
    with st.spinner("Loading data..."):
        df = load_master_data()
        data_summary = get_data_summary(df)
        feature_groups = get_feature_groups(df)

    if df.empty:
        st.error("No data available. Please ensure the master dataset is accessible.")
        st.stop()

    # Check if target column exists
    if TARGET not in df.columns:
        st.error(f"Target column '{TARGET}' not found in dataset.")
        st.stop()

    # Data overview section
    st.markdown("### Data Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{data_summary['total_rows']:,}")

    with col2:
        st.metric("Features", data_summary['num_features'])

    with col3:
        years = (data_summary['date_range'][1] - data_summary['date_range'][0]).days / 365.25
        st.metric("Time Span", f"{years:.1f} years")

    with col4:
        completeness = (1 - data_summary['missing_values'] / (data_summary['total_rows'] * data_summary['num_features'])) * 100
        st.metric("Completeness", f"{completeness:.1f}%")

    st.divider()

    # Time series exploration
    st.markdown(f"### {TARGET_LABEL} Time Series")

    # Date range selector
    col1, col2 = st.columns([2, 1])

    with col1:
        date_filter = st.selectbox(
            "Select time period:",
            get_date_range_options(),
            index=1,
            help="Choose a predefined time period or select custom range"
        )

    with col2:
        if date_filter == "Custom range":
            st.date_input("Start date", df.index.min())
            st.date_input("End date", df.index.max())

    # Apply filter
    filtered_df = apply_date_filter(df, date_filter)

    # Display time series
    st.markdown(f"#### {TARGET_LABEL} - {date_filter}")

    if not filtered_df.empty and TARGET in filtered_df.columns:
        fig = create_time_series_plot(
            filtered_df[[TARGET]],
            [TARGET],
            title=f"Hourly {TARGET_LABEL} ({date_filter})",
            yaxis_title=f"{TARGET_LABEL} ({UNIT})"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        target_data = filtered_df[TARGET]

        with col1:
            st.metric("Mean", f"{target_data.mean():.2f} {UNIT}")
        with col2:
            st.metric("Median", f"{target_data.median():.2f} {UNIT}")
        with col3:
            st.metric("Std Dev", f"{target_data.std():.2f} {UNIT}")
        with col4:
            st.metric("Min", f"{target_data.min():.2f} {UNIT}")
        with col5:
            st.metric("Max", f"{target_data.max():.2f} {UNIT}")

    st.divider()

    # Pattern analysis
    st.markdown(f"### {TARGET_LABEL.split()[0]} Patterns")

    tab1, tab2, tab3 = st.tabs(["📅 Hourly Patterns", "🗓️ Daily Patterns", "📊 Distribution"])

    with tab1:
        st.markdown("#### Average by Hour of Day")

        if TARGET in filtered_df.columns:
            hourly = filtered_df.groupby(filtered_df.index.hour)[TARGET].agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('min', 'min'),
                ('max', 'max'),
                ('median', 'median')
            ])

            if not hourly.empty:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly.index,
                    y=hourly['mean'],
                    mode='lines+markers',
                    name=f'Mean {TARGET_LABEL}',
                    line=dict(color=COLOR, width=3),
                    marker=dict(size=8)
                ))
                fig.add_trace(go.Scatter(
                    x=list(hourly.index) + list(hourly.index[::-1]),
                    y=list(hourly['mean'] + hourly['std']) + list((hourly['mean'] - hourly['std'])[::-1]),
                    fill='toself',
                    fillcolor=f'rgba{tuple(int(COLOR.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Std Deviation',
                    showlegend=True
                ))
                fig.update_layout(
                    title=f"Average Hourly {TARGET_LABEL} with Standard Deviation",
                    xaxis_title="Hour of Day",
                    yaxis_title=f"{TARGET_LABEL} ({UNIT})",
                    height=500,
                    template='plotly_white',
                    xaxis=dict(tickmode='linear', tick0=0, dtick=1)
                )
                st.plotly_chart(fig, use_container_width=True)

                # Key insights
                peak_hour = hourly['mean'].idxmax()
                peak_value = hourly.loc[peak_hour, 'mean']
                low_hour = hourly['mean'].idxmin()
                low_value = hourly.loc[low_hour, 'mean']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak Hour", f"{peak_hour}:00", f"{peak_value:.2f} {UNIT}")
                with col2:
                    st.metric("Lowest Hour", f"{low_hour}:00", f"{low_value:.2f} {UNIT}")
                with col3:
                    variation = ((peak_value - low_value) / low_value) * 100
                    st.metric("Daily Variation", f"{variation:.1f}%")

    with tab2:
        st.markdown("#### Average by Day of Week")

        if TARGET in filtered_df.columns:
            daily = filtered_df.groupby(filtered_df.index.dayofweek)[TARGET].agg([
                ('mean', 'mean'),
                ('std', 'std')
            ])
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                        4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            daily.index = daily.index.map(day_names)

            if not daily.empty:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=daily.index,
                    y=daily['mean'],
                    error_y=dict(type='data', array=daily['std'], visible=True),
                    marker_color=COLOR
                ))
                fig.update_layout(
                    title=f"Average Daily {TARGET_LABEL} with Standard Deviation",
                    xaxis_title="Day of Week",
                    yaxis_title=f"{TARGET_LABEL} ({UNIT})",
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

                # Key insights
                peak_day = daily['mean'].idxmax()
                peak_day_value = daily.loc[peak_day, 'mean']
                low_day = daily['mean'].idxmin()
                low_day_value = daily.loc[low_day, 'mean']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Highest Day", peak_day, f"{peak_day_value:.2f} {UNIT}")
                with col2:
                    st.metric("Lowest Day", low_day, f"{low_day_value:.2f} {UNIT}")

                # Weekend vs Weekday
                weekday_avg = daily.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'mean'].mean()
                weekend_avg = daily.loc[['Saturday', 'Sunday'], 'mean'].mean()
                diff_pct = ((weekday_avg - weekend_avg) / weekend_avg) * 100

                st.info(f"""
                **Weekday vs Weekend:**
                - Weekday average: {weekday_avg:.2f} {UNIT}
                - Weekend average: {weekend_avg:.2f} {UNIT}
                - Difference: {abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'} on weekdays
                """)

    with tab3:
        st.markdown("#### Distribution Analysis")

        if TARGET in filtered_df.columns:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribution (Histogram)", "Box Plot by Month")
            )

            # Histogram
            fig.add_trace(
                go.Histogram(x=filtered_df[TARGET], nbinsx=50, marker_color=COLOR, name='Distribution'),
                row=1, col=1
            )

            # Box plot by month
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month in sorted(filtered_df.index.month.unique()):
                month_data = filtered_df[filtered_df.index.month == month][TARGET]
                fig.add_trace(
                    go.Box(y=month_data, name=month_names[month - 1], marker_color=COLOR, showlegend=False),
                    row=1, col=2
                )

            fig.update_layout(height=450, template='plotly_white', showlegend=False)
            fig.update_xaxes(title_text=f"{TARGET_LABEL} ({UNIT})", row=1, col=1)
            fig.update_yaxes(title_text="Frequency", row=1, col=1)
            fig.update_xaxes(title_text="Month", row=1, col=2)
            fig.update_yaxes(title_text=f"{TARGET_LABEL} ({UNIT})", row=1, col=2)

            st.plotly_chart(fig, use_container_width=True)

            # Percentiles
            st.markdown("#### Percentiles")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("10th", f"{filtered_df[TARGET].quantile(0.10):.2f} {UNIT}")
            with col2:
                st.metric("25th", f"{filtered_df[TARGET].quantile(0.25):.2f} {UNIT}")
            with col3:
                st.metric("50th", f"{filtered_df[TARGET].quantile(0.50):.2f} {UNIT}")
            with col4:
                st.metric("75th", f"{filtered_df[TARGET].quantile(0.75):.2f} {UNIT}")
            with col5:
                st.metric("90th", f"{filtered_df[TARGET].quantile(0.90):.2f} {UNIT}")

    st.divider()

    # Feature analysis
    st.markdown("### Feature Analysis")

    # Get relevant features based on model type
    if MODEL_TYPE == 'consumption':
        relevant_features = [col for col in df.columns if 'consumption' in col.lower() or 'temp' in col.lower() or 'weather' in col.lower()]
    else:
        relevant_features = [col for col in df.columns if 'price' in col.lower() or 'ptf' in col.lower() or 'smf' in col.lower()]

    st.markdown(f"**{len(relevant_features)} related features** found in the dataset.")

    # Feature group selector
    selected_group = st.selectbox(
        "Select feature group to explore:",
        list(feature_groups.keys()),
        help="Choose a category of features to analyze"
    )

    if selected_group in feature_groups:
        group_features = feature_groups[selected_group]

        st.markdown(f"#### {selected_group}")
        st.markdown(f"**{len(group_features)} features** in this group")

        if group_features:
            with st.expander(f"View all features in {selected_group}"):
                cols = st.columns(3)
                for i, feature in enumerate(sorted(group_features)):
                    with cols[i % 3]:
                        st.markdown(f"- `{feature}`")

            st.markdown("##### Visualize Features")
            selected_features = st.multiselect(
                "Select features to visualize:",
                group_features,
                default=group_features[:min(3, len(group_features))],
                max_selections=5,
                help="Choose up to 5 features to plot together"
            )

            if selected_features:
                available_features = [f for f in selected_features if f in filtered_df.columns]

                if available_features:
                    fig = create_time_series_plot(
                        filtered_df[available_features],
                        available_features,
                        title=f"{selected_group} - Selected Features Over Time",
                        yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    if len(available_features) > 1:
                        st.markdown("##### Feature Correlations")
                        corr_features = available_features + [TARGET] if TARGET not in available_features else available_features
                        corr_matrix = filtered_df[corr_features].corr()

                        fig = create_correlation_heatmap(corr_matrix, title=f"Correlation Matrix - {selected_group}")
                        st.plotly_chart(fig, use_container_width=True)

                        if TARGET in corr_matrix.index:
                            target_corr = corr_matrix[TARGET].drop(TARGET).abs().sort_values(ascending=False)
                            st.markdown(f"**Strongest correlations with {TARGET_LABEL}:**")
                            for i, (feature, corr_val) in enumerate(target_corr.head(5).items(), 1):
                                st.markdown(f"{i}. `{feature}`: {corr_val:.3f}")

    st.divider()

    # Data quality
    st.markdown("### Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Missing Values")
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({
            'Feature': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if not missing_df.empty:
            st.dataframe(missing_df.head(20), use_container_width=True, hide_index=True)
        else:
            st.success("No missing values in the dataset!")

    with col2:
        st.markdown("#### Data Statistics")
        st.metric("Data Completeness", f"{completeness:.2f}%")
        st.metric("Total Features", data_summary['num_features'])
        st.metric("Total Records", f"{data_summary['total_rows']:,}")

        start_date = data_summary['date_range'][0].strftime('%Y-%m-%d')
        end_date = data_summary['date_range'][1].strftime('%Y-%m-%d')
        st.markdown(f"**Date Range:** {start_date} to {end_date}")

    st.divider()

    # Download section
    st.markdown("### Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Download Filtered Data (CSV)", use_container_width=True):
            csv = filtered_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"forewatt_{MODEL_TYPE}_data_{date_filter.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Download Summary Statistics", use_container_width=True):
            summary_stats = filtered_df.describe()
            csv = summary_stats.to_csv()
            st.download_button(
                label="Download Statistics CSV",
                data=csv,
                file_name=f"forewatt_{MODEL_TYPE}_statistics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
