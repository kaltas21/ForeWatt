"""
ForeWatt Dashboard - Data Explorer Page
Explore historical data, features, and consumption patterns.
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
    apply_date_filter, get_date_range_options, get_hourly_patterns,
    get_daily_patterns, TARGET_VARIABLE, PAGE_CONFIG,
    create_time_series_plot, create_correlation_heatmap,
    create_hourly_pattern_plot, create_box_plot
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

st.title("📊 Data Explorer")
st.markdown("Explore historical electricity consumption data, features, and patterns.")

st.divider()

try:
    # Load data
    with st.spinner("Loading data..."):
        df = load_master_data()
        data_summary = get_data_summary(df)
        feature_groups = get_feature_groups(df)

    if df.empty:
        st.error("No data available. Please ensure the master dataset is accessible.")
        st.stop()

    # Data overview section
    st.markdown("## Data Overview")

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
    st.markdown("## Time Series Analysis")

    # Date range selector
    col1, col2 = st.columns([2, 1])

    with col1:
        date_filter = st.selectbox(
            "Select time period:",
            get_date_range_options(),
            index=1,  # Last 7 days
            help="Choose a predefined time period or select custom range"
        )

    with col2:
        if date_filter == "Custom range":
            st.date_input("Start date", df.index.min())
            st.date_input("End date", df.index.max())

    # Apply filter
    filtered_df = apply_date_filter(df, date_filter)

    # Display consumption time series
    st.markdown(f"### Electricity Consumption - {date_filter}")

    if not filtered_df.empty and TARGET_VARIABLE in filtered_df.columns:
        fig = create_time_series_plot(
            filtered_df[[TARGET_VARIABLE]],
            [TARGET_VARIABLE],
            title=f"Hourly Electricity Consumption ({date_filter})",
            yaxis_title="Consumption (MWh)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3, col4, col5 = st.columns(5)

        consumption = filtered_df[TARGET_VARIABLE]

        with col1:
            st.metric("Mean", f"{consumption.mean():.0f} MWh")

        with col2:
            st.metric("Median", f"{consumption.median():.0f} MWh")

        with col3:
            st.metric("Std Dev", f"{consumption.std():.0f} MWh")

        with col4:
            st.metric("Min", f"{consumption.min():.0f} MWh")

        with col5:
            st.metric("Max", f"{consumption.max():.0f} MWh")

    st.divider()

    # Pattern analysis
    st.markdown("## Consumption Patterns")

    tab1, tab2 = st.tabs(["📅 Hourly Patterns", "🗓️ Daily Patterns"])

    with tab1:
        st.markdown("### Average Consumption by Hour of Day")

        hourly_patterns = get_hourly_patterns(filtered_df)

        if not hourly_patterns.empty:
            fig = create_hourly_pattern_plot(
                hourly_patterns,
                title="Average Hourly Consumption with Standard Deviation"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Key insights
            peak_hour = hourly_patterns['mean'].idxmax()
            peak_value = hourly_patterns.loc[peak_hour, 'mean']
            low_hour = hourly_patterns['mean'].idxmin()
            low_value = hourly_patterns.loc[low_hour, 'mean']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Peak Hour", f"{peak_hour}:00", f"{peak_value:.0f} MWh")

            with col2:
                st.metric("Lowest Hour", f"{low_hour}:00", f"{low_value:.0f} MWh")

            with col3:
                variation = ((peak_value - low_value) / low_value) * 100
                st.metric("Daily Variation", f"{variation:.1f}%")

    with tab2:
        st.markdown("### Average Consumption by Day of Week")

        daily_patterns = get_daily_patterns(filtered_df)

        if not daily_patterns.empty:
            import plotly.graph_objects as go

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=daily_patterns.index,
                y=daily_patterns['mean'],
                error_y=dict(
                    type='data',
                    array=daily_patterns['std'],
                    visible=True
                ),
                marker_color='#1f77b4'
            ))

            fig.update_layout(
                title="Average Daily Consumption with Standard Deviation",
                xaxis_title="Day of Week",
                yaxis_title="Consumption (MWh)",
                height=500,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Key insights
            peak_day = daily_patterns['mean'].idxmax()
            peak_day_value = daily_patterns.loc[peak_day, 'mean']
            low_day = daily_patterns['mean'].idxmin()
            low_day_value = daily_patterns.loc[low_day, 'mean']

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Highest Day", peak_day, f"{peak_day_value:.0f} MWh")

            with col2:
                st.metric("Lowest Day", low_day, f"{low_day_value:.0f} MWh")

            # Weekend vs Weekday comparison
            weekday_avg = daily_patterns.loc[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], 'mean'].mean()
            weekend_avg = daily_patterns.loc[['Saturday', 'Sunday'], 'mean'].mean()
            diff_pct = ((weekday_avg - weekend_avg) / weekend_avg) * 100

            st.info(f"""
            **Weekday vs Weekend:**
            - Weekday average: {weekday_avg:.0f} MWh
            - Weekend average: {weekend_avg:.0f} MWh
            - Difference: {abs(diff_pct):.1f}% {'higher' if diff_pct > 0 else 'lower'} on weekdays
            """)


    st.divider()

    # Feature analysis
    st.markdown("## Feature Analysis")

    st.markdown("""
    The dataset contains **106 engineered features** grouped into categories:
    lag features, rolling statistics, calendar features, weather data, prices, and macroeconomic indicators.
    """)

    # Feature group selector
    selected_group = st.selectbox(
        "Select feature group to explore:",
        list(feature_groups.keys()),
        help="Choose a category of features to analyze"
    )

    if selected_group in feature_groups:
        group_features = feature_groups[selected_group]

        st.markdown(f"### {selected_group}")
        st.markdown(f"**{len(group_features)} features** in this group")

        if group_features:
            # Show feature list
            with st.expander(f"View all features in {selected_group}"):
                cols = st.columns(3)
                for i, feature in enumerate(sorted(group_features)):
                    with cols[i % 3]:
                        st.markdown(f"- `{feature}`")

            # Feature visualization
            st.markdown("#### Visualize Features")

            # Select up to 5 features to plot
            selected_features = st.multiselect(
                "Select features to visualize:",
                group_features,
                default=group_features[:min(3, len(group_features))],
                max_selections=5,
                help="Choose up to 5 features to plot together"
            )

            if selected_features:
                # Plot selected features
                available_features = [f for f in selected_features if f in filtered_df.columns]

                if available_features:
                    fig = create_time_series_plot(
                        filtered_df[available_features],
                        available_features,
                        title=f"{selected_group} - Selected Features Over Time",
                        yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Correlation analysis
                    if len(available_features) > 1:
                        st.markdown("#### Feature Correlations")

                        corr_features = available_features + [TARGET_VARIABLE] if TARGET_VARIABLE not in available_features else available_features
                        corr_matrix = filtered_df[corr_features].corr()

                        fig = create_correlation_heatmap(
                            corr_matrix,
                            title=f"Correlation Matrix - {selected_group}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Highlight strongest correlations with target
                        if TARGET_VARIABLE in corr_matrix.index:
                            target_corr = corr_matrix[TARGET_VARIABLE].drop(TARGET_VARIABLE).abs().sort_values(ascending=False)

                            st.markdown("**Strongest correlations with consumption:**")
                            for i, (feature, corr_val) in enumerate(target_corr.head(5).items(), 1):
                                st.markdown(f"{i}. `{feature}`: {corr_val:.3f}")

    st.divider()

    # Data quality
    st.markdown("## Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Missing Values")

        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100

        missing_df = pd.DataFrame({
            'Feature': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing %': missing_pct.values
        })

        # Filter to show only features with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        missing_df = missing_df.sort_values('Missing Count', ascending=False)

        if not missing_df.empty:
            st.dataframe(
                missing_df.head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✅ No missing values in the dataset!")

    with col2:
        st.markdown("### Data Statistics")

        st.metric(
            "Data Completeness",
            f"{(1 - data_summary['missing_values'] / (data_summary['total_rows'] * data_summary['num_features'])) * 100:.2f}%"
        )

        st.metric("Total Features", data_summary['num_features'])
        st.metric("Total Records", f"{data_summary['total_rows']:,}")

        # Date range
        start_date = data_summary['date_range'][0].strftime('%Y-%m-%d')
        end_date = data_summary['date_range'][1].strftime('%Y-%m-%d')
        st.markdown(f"**Date Range:**  \n{start_date} to {end_date}")

    st.divider()

    # Download section
    st.markdown("## Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download Filtered Data (CSV)", use_container_width=True):
            csv = filtered_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"forewatt_data_{date_filter.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📊 Download Summary Statistics", use_container_width=True):
            summary_stats = filtered_df.describe()
            csv = summary_stats.to_csv()
            st.download_button(
                label="Download Statistics CSV",
                data=csv,
                file_name=f"forewatt_statistics_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
