"""
ForeWatt Dashboard - Model Comparison Page
Compare all trained models with detailed metrics and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    PAGE_CONFIG,
    load_all_metrics, get_best_models_per_type, get_available_models,
    load_feature_importance, create_feature_importance_plot
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)

st.title("🔬 Model Comparison")
st.markdown("Compare all trained models across multiple metrics and configurations.")

st.divider()

# Sidebar controls
st.sidebar.header("Comparison Settings")

# Target selection
target = st.sidebar.selectbox(
    "Target Variable",
    ["consumption", "price_real"],
    index=0,
    help="Select the target variable"
)

# Model category filter
model_category = st.sidebar.selectbox(
    "Model Category",
    ["All", "Baseline", "Deep Learning"],
    index=0,
    help="Filter models by category"
)

category_map = {
    "All": None,
    "Baseline": "baseline",
    "Deep Learning": "deeplearning"
}

try:
    # Load all metrics
    with st.spinner("Loading model results..."):
        df_all = load_all_metrics(target=target, model_category=category_map[model_category])

    if df_all.empty:
        st.warning(f"No model results found for target={target} and category={model_category}")
        st.info("""
        Please ensure:
        1. Models have been trained
        2. Results are saved in `/reports/new_experiment/`
        3. The target variable matches the training configuration
        """)
        st.stop()

    # Overview metrics
    st.markdown("## 📊 Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Models",
            f"{len(df_all):,}",
            help="Total number of trained model configurations"
        )

    with col2:
        best_mae = df_all['MAE'].min()
        st.metric(
            "Best MAE",
            f"{best_mae:.2f}",
            help="Lowest Mean Absolute Error achieved"
        )

    with col3:
        best_mase = df_all['MASE'].min()
        st.metric(
            "Best MASE",
            f"{best_mase:.3f}",
            help="Lowest Mean Absolute Scaled Error achieved"
        )

    with col4:
        models_under_1 = len(df_all[df_all['MASE'] < 1.0])
        st.metric(
            "MASE < 1.0",
            f"{models_under_1}",
            help="Models beating naive baseline"
        )

    st.divider()

    # Best models per type
    st.markdown("## 🏆 Best Models by Type")

    best_models = get_best_models_per_type(target=target, metric='MAE')

    if not best_models.empty:
        # Display formatted table
        display_cols = ['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                       'category', 'n_features', 'feature_tier']

        display_df = best_models[display_cols].copy()
        display_df.columns = ['Model', 'Configuration', 'MAE', 'sMAPE', 'MASE',
                             'Category', '# Features', 'Feature Tier']

        # Format numeric columns
        display_df['MAE'] = display_df['MAE'].apply(lambda x: f"{x:.2f}")
        display_df['sMAPE'] = display_df['sMAPE'].apply(lambda x: f"{x:.2f}")
        display_df['MASE'] = display_df['MASE'].apply(lambda x: f"{x:.3f}")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Best model highlight
        best_idx = best_models['MAE'].idxmin()
        best_model = best_models.loc[best_idx]

        st.success(f"""
        🏆 **Best Overall Model:** {best_model['model_name']} ({best_model['config_name']})
        - **MAE:** {best_model['MAE']:.2f}
        - **MASE:** {best_model['MASE']:.3f}
        - **Feature Tier:** {best_model['feature_tier']}
        """)

    st.divider()

    # Detailed comparison tabs
    st.markdown("## 📈 Detailed Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 All Configurations",
        "🎯 Feature Tiers",
        "📉 Performance Distribution",
        "🔍 Feature Importance"
    ])

    with tab1:
        st.markdown("### All Model Configurations")

        # Sorting options
        sort_by = st.selectbox(
            "Sort by",
            ['MAE', 'sMAPE', 'MASE', 'val_MAE', 'training_time'],
            index=0
        )

        # Display all models
        display_all_df = df_all[['model_name', 'config_name', 'MAE', 'sMAPE', 'MASE',
                                 'val_MAE', 'val_MASE', 'n_features', 'feature_tier',
                                 'training_time', 'category']].copy()

        display_all_df = display_all_df.sort_values(sort_by)

        # Format columns
        for col in ['MAE', 'sMAPE', 'val_MAE']:
            if col in display_all_df.columns:
                display_all_df[col] = display_all_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

        for col in ['MASE', 'val_MASE']:
            if col in display_all_df.columns:
                display_all_df[col] = display_all_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")

        if 'training_time' in display_all_df.columns:
            display_all_df['training_time'] = display_all_df['training_time'].apply(
                lambda x: f"{x:.2f}s" if pd.notnull(x) else "N/A"
            )

        st.dataframe(display_all_df, use_container_width=True, height=400)

        # Download button
        csv = df_all.to_csv(index=False)
        st.download_button(
            label="📥 Download full results (CSV)",
            data=csv,
            file_name=f"model_comparison_{target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    with tab2:
        st.markdown("### Performance by Feature Tier")

        if 'feature_tier' in df_all.columns:
            tier_summary = df_all.groupby('feature_tier').agg({
                'MAE': ['mean', 'min', 'count'],
                'MASE': ['mean', 'min'],
                'n_features': 'first'
            }).round(3)

            tier_summary.columns = ['MAE (avg)', 'MAE (best)', 'Count',
                                   'MASE (avg)', 'MASE (best)', 'Features']

            st.dataframe(tier_summary, use_container_width=True)

            # Visualization
            import plotly.express as px

            fig = px.box(
                df_all,
                x='feature_tier',
                y='MAE',
                color='model_name',
                title="MAE Distribution by Feature Tier",
                labels={'feature_tier': 'Feature Tier', 'MAE': 'Mean Absolute Error'}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature tier information not available in the results.")

    with tab3:
        st.markdown("### Performance Distribution")

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE Distribution', 'MASE Distribution',
                          'sMAPE Distribution', 'Training Time Distribution')
        )

        # MAE histogram
        fig.add_trace(
            go.Histogram(x=df_all['MAE'], name='MAE', nbinsx=30),
            row=1, col=1
        )

        # MASE histogram
        fig.add_trace(
            go.Histogram(x=df_all['MASE'], name='MASE', nbinsx=30),
            row=1, col=2
        )

        # sMAPE histogram
        fig.add_trace(
            go.Histogram(x=df_all['sMAPE'], name='sMAPE', nbinsx=30),
            row=2, col=1
        )

        # Training time histogram
        if 'training_time' in df_all.columns:
            fig.add_trace(
                go.Histogram(x=df_all['training_time'], name='Training Time (s)', nbinsx=30),
                row=2, col=2
            )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Feature Importance Analysis")

        # Model selection for feature importance
        available_configs = df_all[df_all['model_name'].isin(['CATBOOST', 'XGBOOST', 'LIGHTGBM'])].copy()

        if not available_configs.empty:
            # Select best model of each type
            best_configs = available_configs.loc[available_configs.groupby('model_name')['MAE'].idxmin()]

            selected_model = st.selectbox(
                "Select model to view feature importance:",
                best_configs['model_name'].tolist()
            )

            if selected_model:
                # Get config hash for selected model
                model_row = best_configs[best_configs['model_name'] == selected_model].iloc[0]
                config_hash = model_row['config_hash']
                model_type = model_row['model_type']

                # Load feature importance
                fi_df = load_feature_importance(model_type, config_hash, target)

                if fi_df is not None and not fi_df.empty:
                    st.markdown(f"#### {selected_model} - Top 20 Features")

                    # Show top features
                    top_features = fi_df.head(20)

                    # Create bar chart - pass DataFrame directly
                    fig = create_feature_importance_plot(
                        top_features,  # Pass DataFrame directly
                        top_n=20,
                        title=f"Top 20 Features - {selected_model}"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Show table
                    st.dataframe(top_features, use_container_width=True, hide_index=True)
                else:
                    st.warning(f"Feature importance not available for {selected_model}")
        else:
            st.info("Feature importance is only available for tree-based models (CatBoost, XGBoost, LightGBM)")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
