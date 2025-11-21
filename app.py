
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import scikitplot as skplt
import os

# Set page configuration
st.set_page_config(
    page_title="Advanced Income Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-high {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 6px 10px rgba(0,0,0,0.15);
    }
    .prediction-low {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 6px 10px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models and metadata"""
    try:
        enhanced_info = joblib.load('enhanced_model_info.pkl')
        models = {}
        
        # Load all individual models
        for model_name in enhanced_info['all_models']:
            clean_name = model_name.lower().replace(' ', '_')
            model_path = f'model_{clean_name}.pkl'
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        
        # Load best model
        best_model = joblib.load('best_model.pkl')
        
        return models, best_model, enhanced_info
        
    except FileNotFoundError:
        st.error("""
         Model files not found! Please run `train_model.py` first.
        
        Steps:
        1. Place your CSV file in the same directory
        2. Update CSV_FILE_PATH in train_model.py
        3. Run: python train_model.py
        4. Wait for training to complete (this will take a while)
        5. Refresh this app
        """)
        return None, None, None

def create_advanced_input_form(model_info):
    """Create advanced input form with sections"""
    input_data = {}
    
    # Personal Information Section
    st.header("Personal Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'age' in model_info['numerical_columns']:
            input_data['age'] = st.slider("Age", 0, 100, 35, help="Age of the individual")
        
        if 'gender' in model_info['categorical_columns']:
            input_data['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])
    
    with col2:
        if 'education' in model_info['categorical_columns']:
            input_data['education'] = st.selectbox("Education Level", [
                "High school graduate", "12th grade no diploma", "Children", 
                "Bachelors", "Masters", "Doctorate", "Associates", "Some college",
                "Professional degree", "1st-4th grade", "5th-6th grade", 
                "7th-8th grade", "9th grade", "10th grade", "11th grade"
            ])
    
    with col3:
        if 'marital_status' in model_info['categorical_columns']:
            input_data['marital_status'] = st.selectbox("Marital Status", [
                "Married-civilian spouse present", "Never married", "Widowed", 
                "Divorced", "Separated", "Married-spouse absent"
            ])
        
        if 'race' in model_info['categorical_columns']:
            input_data['race'] = st.selectbox("Race", [
                "White", "Black", "Asian or Pacific Islander", 
                "Amer Indian Aleut or Eskimo", "Other"
            ])
    
    # Employment Section
    st.header("Employment Information")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'employment_commitment' in model_info['categorical_columns']:
            input_data['employment_commitment'] = st.selectbox("Employment Status", [
                "Not in labor force", "Children or Armed Forces", "Full-time schedules", 
                "Part-time schedules", "Unemployed full-time", "Unemployed part-time"
            ])
        
        if 'class' in model_info['categorical_columns']:
            input_data['class'] = st.selectbox("Employment Class", [
                "Federal government", "Local government", "State government", 
                "Self-employed incorporated", "Self-employed not incorporated", 
                "Private for-profit", "Private not-for-profit", "Without pay"
            ])
    
    with col2:
        if 'is_hispanic' in model_info['categorical_columns']:
            input_data['is_hispanic'] = st.selectbox("Hispanic Origin", [
                "All other", "Mexican American", "Mexican", "Puerto Rican", 
                "Cuban", "Central or South American", "Other Spanish", "Chicano"
            ])
        
        if 'country_of_birth_self' in model_info['categorical_columns']:
            input_data['country_of_birth_self'] = st.selectbox("Country of Birth", [
                "US", "Mexico", "Puerto Rico", "Cuba", "Dominican Republic",
                "Jamaica", "India", "China", "Philippines", "Vietnam",
                "El Salvador", "Canada", "United Kingdom", "Germany", "Other"
            ])
    
    # Additional Features Section
    st.header("📈 Additional Features")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'importance_of_record' in model_info['numerical_columns']:
            input_data['importance_of_record'] = st.number_input(
                "Importance of Record", 
                min_value=0.0, 
                max_value=10000.0, 
                value=1000.0,
                step=100.0
            )
    
    with col2:
        # Add any other important numerical features
        other_numerical = [col for col in model_info['numerical_columns'] 
                          if col not in ['age', 'importance_of_record']]
        for col in other_numerical[:2]:  # Show first 2 additional numerical features
            input_data[col] = st.number_input(f"{col.replace('_', ' ').title()}", value=0.0)
    
    # Fill remaining columns with default values
    for col in model_info['all_columns']:
        if col not in input_data:
            if col in model_info['numerical_columns']:
                input_data[col] = 0.0
            else:
                input_data[col] = "Unknown"
    
    return input_data

def plot_advanced_metrics(selected_model, model_name, X_test, y_test):
    """Create advanced metric visualizations for a selected model"""
    if selected_model is None:
        return
    
    # Get predictions
    y_pred = selected_model.predict(X_test)
    y_pred_proba = selected_model.predict_proba(X_test)[:, 1]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'Confusion Matrix - {model_name}',
            f'ROC Curve - {model_name}',
            f'Precision-Recall Curve - {model_name}',
            f'Feature Importance - {model_name}'
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "bar"}]
        ]
    )
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=['Below', 'Above'],
            y=['Below', 'Above'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ),
        row=1, col=1
    )
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=3)
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=2
    )
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color='green', width=3)
        ),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Recall", row=2, col=1)
    fig.update_yaxes(title_text="Precision", row=2, col=1)
    
    # 4. Feature Importance (if available)
    try:
        if hasattr(selected_model.named_steps['classifier'], 'feature_importances_'):
            importances = selected_model.named_steps['classifier'].feature_importances_
            feature_names = selected_model.named_steps['preprocessor'].get_feature_names_out()
            
            # Get top 10 features
            indices = np.argsort(importances)[::-1][:10]
            
            fig.add_trace(
                go.Bar(
                    x=importances[indices],
                    y=[feature_names[i] for i in indices],
                    orientation='h',
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
            fig.update_xaxes(title_text="Importance", row=2, col=2)
    except:
        fig.add_annotation(
            text="Feature importance not available",
            x=0.5, y=0.5,
            xref="x4", yref="y4",
            showarrow=False,
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text=f"Advanced Model Analysis - {model_name}")
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<div class="main-header"> Advanced Income Inequality Predictor</div>', unsafe_allow_html=True)
    st.write("Multi-Model Analysis with Comprehensive Visualizations")
    
    # Load models
    models, best_model, enhanced_info = load_models()
    
    if models is None:
        return
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Dashboard", 
        " Model Comparison", 
        " Single Prediction", 
        " Batch Analysis", 
        " Advanced Analytics"
    ])
    
    with tab1:
        st.header(" Model Performance Dashboard")
        
        # Performance metrics
        st.subheader(" Model Performance Summary")
        performance_data = enhanced_info['model_performance']
        
        # Create metrics columns
        cols = st.columns(4)
        best_model_name = enhanced_info['best_model']
        best_f1 = performance_data[best_model_name]['f1_score']
        
        with cols[0]:
            st.metric("Best Model", best_model_name)
        with cols[1]:
            st.metric("Best F1 Score", f"{best_f1:.4f}")
        with cols[2]:
            st.metric("Total Models", len(enhanced_info['all_models']))
        with cols[3]:
            st.metric("Dataset Features", len(enhanced_info['all_columns']))
        
        # Performance comparison chart
        st.subheader(" Model Performance Comparison")
        model_names = list(performance_data.keys())
        f1_scores = [performance_data[name]['f1_score'] for name in model_names]
        accuracies = [performance_data[name]['accuracy'] for name in model_names]
        auc_scores = [performance_data[name]['roc_auc'] for name in model_names]
        
        fig = go.Figure(data=[
            go.Bar(name='F1 Score', x=model_names, y=f1_scores, marker_color='skyblue'),
            go.Bar(name='Accuracy', x=model_names, y=accuracies, marker_color='lightgreen'),
            go.Bar(name='ROC AUC', x=model_names, y=auc_scores, marker_color='salmon')
        ])
        fig.update_layout(barmode='group', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show saved visualizations
        st.subheader(" Model Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('model_comparison.png'):
                st.image('model_comparison.png', caption='Model Comparison', use_column_width=True)
        
        with col2:
            if os.path.exists('roc_curves.png'):
                st.image('roc_curves.png', caption='ROC Curves', use_column_width=True)
    
    with tab2:
        st.header("Detailed Model Comparison")
        
        # Model selection for detailed analysis
        selected_model = st.selectbox(
            "Select Model for Detailed Analysis",
            enhanced_info['all_models']
        )
        
        if selected_model:
            model_details = enhanced_info['model_performance'][selected_model]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("F1 Score", f"{model_details['f1_score']:.4f}")
            with col2:
                st.metric("Accuracy", f"{model_details['accuracy']:.4f}")
            with col3:
                st.metric("ROC AUC", f"{model_details['roc_auc']:.4f}")
            with col4:
                st.metric("Status", " Best" if selected_model == enhanced_info['best_model'] else "⚡ Good")
            
            # Show best parameters
            st.subheader(" Best Hyperparameters")
            st.json(model_details['best_params'])
            
            # Show confusion matrix for selected model
            if os.path.exists('confusion_matrices.png'):
                st.image('confusion_matrices.png', caption='Confusion Matrices for Top 3 Models', use_column_width=True)
    
    with tab3:
        st.header(" Single Prediction")
        
        # Model selection for prediction
        prediction_model = st.selectbox(
            "Select Prediction Model",
            enhanced_info['all_models'],
            index=enhanced_info['all_models'].index(enhanced_info['best_model']) if enhanced_info['best_model'] in enhanced_info['all_models'] else 0
        )
        
        # Create input form
        input_data = create_advanced_input_form(enhanced_info)
        
        if st.button("Predict Income Level", type="primary", use_container_width=True):
            with st.spinner("Running advanced prediction..."):
                try:
                    input_df = pd.DataFrame([input_data])
                    input_df = input_df[enhanced_info['all_columns']]
                    
                    selected_ml_model = models[prediction_model]
                    prediction = selected_ml_model.predict(input_df)[0]
                    probabilities = selected_ml_model.predict_proba(input_df)[0]
                    
                    # Enhanced result display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-high">
                                <h2> PREDICTION: ABOVE INCOME LIMIT</h2>
                                <p style="font-size: 1.5em; margin: 20px 0;">
                                Confidence: {probabilities[1]:.2%}
                                </p>
                                <p>Model Used: {prediction_model}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-low">
                                <h2> PREDICTION: BELOW INCOME LIMIT</h2>
                                <p style="font-size: 1.5em; margin: 20px 0;">
                                Confidence: {probabilities[0]:.2%}
                                </p>
                                <p>Model Used: {prediction_model}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Advanced probability visualization
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Below Limit', 'Above Limit'],
                        y=[probabilities[0], probabilities[1]],
                        marker_color=['red', 'green'],
                        text=[f'{probabilities[0]:.2%}', f'{probabilities[1]:.2%}'],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        title='Prediction Probability Distribution',
                        yaxis_title='Probability',
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
    
    with tab4:
        st.header("Batch Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type="csv")
        
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f" Loaded {len(batch_df)} records")
            
            # Model selection for batch prediction
            batch_model = st.selectbox(
                "Select Model for Batch Prediction",
                enhanced_info['all_models'],
                key="batch_model"
            )
            
            if st.button("Run Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    try:
                        # Ensure all columns are present
                        missing_cols = set(enhanced_info['all_columns']) - set(batch_df.columns)
                        if missing_cols:
                            st.error(f"Missing columns: {missing_cols}")
                        else:
                            selected_model = models[batch_model]
                            predictions = selected_model.predict(batch_df)
                            probabilities = selected_model.predict_proba(batch_df)
                            
                            results_df = batch_df.copy()
                            results_df['prediction'] = predictions
                            results_df['probability_below'] = probabilities[:, 0]
                            results_df['probability_above'] = probabilities[:, 1]
                            results_df['income_prediction'] = results_df['prediction'].map({
                                0: 'Below limit', 
                                1: 'Above limit'
                            })
                            results_df['prediction_confidence'] = np.max(probabilities, axis=1)
                            
                            st.subheader("📋 Batch Prediction Results")
                            st.dataframe(results_df)
                            
                            # Enhanced batch analysis
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Records", len(results_df))
                            with col2:
                                above_count = (results_df['prediction'] == 1).sum()
                                st.metric("Above Limit", f"{above_count}")
                            with col3:
                                st.metric("Above Limit %", f"{above_count/len(results_df)*100:.1f}%")
                            with col4:
                                avg_confidence = results_df['prediction_confidence'].mean()
                                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                            
                            # Distribution visualization
                            fig = px.pie(
                                results_df, 
                                names='income_prediction',
                                title='Prediction Distribution',
                                color='income_prediction',
                                color_discrete_map={'Below limit': 'red', 'Above limit': 'green'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                " Download Full Results",
                                csv,
                                "enhanced_batch_predictions.csv",
                                "text/csv"
                            )
                            
                    except Exception as e:
                        st.error(f"Batch processing error: {str(e)}")
    
    with tab5:
        st.header("📈 Advanced Analytics")
        
        st.subheader("Model Performance Deep Dive")
        
        # Show all performance metrics in a table
        performance_list = []
        for model_name, metrics in enhanced_info['model_performance'].items():
            performance_list.append({
                'Model': model_name,
                'F1 Score': metrics['f1_score'],
                'Accuracy': metrics['accuracy'],
                'ROC AUC': metrics['roc_auc'],
                'Is Best': model_name == enhanced_info['best_model']
            })
        
        performance_df = pd.DataFrame(performance_list)
        performance_df = performance_df.sort_values('F1 Score', ascending=False)
        
        st.dataframe(
            performance_df.style.highlight_max(subset=['F1 Score', 'Accuracy', 'ROC AUC']),
            use_container_width=True
        )
        
        # Feature importance analysis
        st.subheader(" Feature Importance Analysis")
        
        # Try to get feature importance from best model
        try:
            best_model_obj = models[enhanced_info['best_model']]
            if hasattr(best_model_obj.named_steps['classifier'], 'feature_importances_'):
                importances = best_model_obj.named_steps['classifier'].feature_importances_
                feature_names = best_model_obj.named_steps['preprocessor'].get_feature_names_out()
                
                # Create feature importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    color='importance',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance not available for the best model")
        
        # Show precision-recall curve visualization
        if os.path.exists('precision_recall_curves.png'):
            st.image('precision_recall_curves.png', caption='Precision-Recall Curves', use_column_width=True)

if __name__ == "__main__":
    main()