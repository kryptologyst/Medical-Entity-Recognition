"""Streamlit demo app for medical entity recognition."""

import streamlit as st
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict
import logging
from pathlib import Path

# Import our modules
from src.models.biobert_model import BioBERTNERModel
from src.utils import get_device, set_seed
from src.utils.deid import DeIdentifier
from src.utils.explainability import ModelExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Medical Entity Recognition Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Disclaimer banner
st.error("""
**IMPORTANT DISCLAIMER**: This is a research demonstration tool only. 
It is NOT intended for clinical use, diagnosis, or medical advice. 
Always consult qualified healthcare professionals for medical decisions.
""")

# Title and description
st.title("🏥 Medical Entity Recognition Demo")
st.markdown("""
This demo showcases a BioBERT-based Named Entity Recognition system for extracting medical entities from clinical text.
The system can identify diseases, chemicals (medications), symptoms, and procedures in clinical notes.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_options = {
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT"
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=0
)

# Privacy settings
st.sidebar.subheader("Privacy Settings")
enable_deid = st.sidebar.checkbox("Enable De-identification", value=True)
show_phi_warning = st.sidebar.checkbox("Show PHI warnings", value=True)

# Initialize de-identifier
deidentifier = DeIdentifier(enable_deid)

# Load model function
@st.cache_resource
def load_model(model_name: str):
    """Load the selected model."""
    try:
        device = get_device("auto")
        model_config = {
            'model_name': model_options[model_name],
            'max_length': 512,
            'num_labels': 9,
            'dropout': 0.1
        }
        
        model = BioBERTNERModel(model_config)
        model.to(device)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Load model
with st.spinner("Loading model..."):
    model, device = load_model(selected_model)

if model is None:
    st.error("Failed to load model. Please try again.")
    st.stop()

# Main interface
st.header("Clinical Text Analysis")

# Input text area
st.subheader("Enter Clinical Text")
sample_texts = {
    "Sample 1": """
    The patient was diagnosed with hypertension and type 2 diabetes mellitus. 
    Treatment started with metformin 500mg twice daily and lisinopril 10mg once daily. 
    Patient reports occasional dizziness and fatigue. Blood pressure was 150/95 mmHg.
    """,
    "Sample 2": """
    Emergency department visit for chest pain and shortness of breath. 
    Patient underwent ECG and chest X-ray. Troponin levels were elevated. 
    Started on aspirin and atorvastatin. Discharged with follow-up in cardiology.
    """,
    "Sample 3": """
    Follow-up visit for chronic kidney disease. Patient is on dialysis three times weekly.
    Current medications include erythropoietin and calcium carbonate. 
    No new symptoms reported. Creatinine level stable at 4.2 mg/dL.
    """
}

selected_sample = st.selectbox("Choose a sample text:", ["Custom"] + list(sample_texts.keys()))

if selected_sample == "Custom":
    input_text = st.text_area(
        "Enter your clinical text here:",
        height=200,
        placeholder="Enter clinical text for entity recognition..."
    )
else:
    input_text = st.text_area(
        "Clinical text:",
        value=sample_texts[selected_sample],
        height=200
    )

# Process button
if st.button("Analyze Text", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # De-identify text if enabled
                if enable_deid:
                    deidentified_text, phi_entities = deidentifier.deidentify_text(input_text)
                    if phi_entities and show_phi_warning:
                        st.warning(f"⚠️ Detected {len(phi_entities)} potential PHI entities. De-identification applied.")
                        with st.expander("View PHI Detection Details"):
                            phi_df = pd.DataFrame(phi_entities)
                            st.dataframe(phi_df)
                    
                    text_to_analyze = deidentified_text
                else:
                    text_to_analyze = input_text
                
                # Run NER prediction
                predictions = model.predict([text_to_analyze])
                entities = predictions[0]
                
                # Display results
                st.subheader("Detected Entities")
                
                if entities:
                    # Create entity dataframe
                    entity_data = []
                    for entity in entities:
                        entity_data.append({
                            'Text': entity['text'],
                            'Type': entity['label'],
                            'Start': entity['start'],
                            'End': entity['end'],
                            'Confidence': f"{entity.get('confidence', 0.0):.3f}"
                        })
                    
                    entity_df = pd.DataFrame(entity_data)
                    st.dataframe(entity_df, use_container_width=True)
                    
                    # Entity type distribution
                    st.subheader("Entity Type Distribution")
                    type_counts = entity_df['Type'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart
                        fig_bar = px.bar(
                            x=type_counts.index,
                            y=type_counts.values,
                            title="Entity Count by Type",
                            labels={'x': 'Entity Type', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    with col2:
                        # Pie chart
                        fig_pie = px.pie(
                            values=type_counts.values,
                            names=type_counts.index,
                            title="Entity Distribution"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Highlighted text
                    st.subheader("Text with Highlighted Entities")
                    highlighted_text = text_to_analyze
                    
                    # Color mapping for entity types
                    colors = {
                        'DISEASE': '#FF6B6B',
                        'CHEMICAL': '#4ECDC4',
                        'SYMPTOM': '#45B7D1',
                        'PROCEDURE': '#96CEB4',
                        'FINDING': '#FFEAA7'
                    }
                    
                    # Sort entities by start position (descending) to avoid index shifts
                    sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
                    
                    for entity in sorted_entities:
                        start = entity['start']
                        end = entity['end']
                        entity_type = entity['label']
                        color = colors.get(entity_type, '#DDA0DD')
                        
                        highlighted_text = (
                            highlighted_text[:start] +
                            f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{entity["text"]}</span>' +
                            highlighted_text[end:]
                        )
                    
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                    
                    # Entity confidence analysis
                    st.subheader("Entity Confidence Analysis")
                    confidence_data = [float(entity['Confidence']) for entity in entity_data]
                    
                    fig_confidence = go.Figure(data=go.Histogram(x=confidence_data, nbinsx=20))
                    fig_confidence.update_layout(
                        title="Distribution of Entity Confidence Scores",
                        xaxis_title="Confidence Score",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_confidence, use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Entities", len(entities))
                    with col2:
                        st.metric("Entity Types", len(type_counts))
                    with col3:
                        st.metric("Avg Confidence", f"{np.mean(confidence_data):.3f}")
                    with col4:
                        st.metric("High Confidence (>0.8)", sum(1 for c in confidence_data if c > 0.8))
                
                else:
                    st.info("No entities detected in the text.")
                
            except Exception as e:
                st.error(f"Error analyzing text: {e}")
                logger.error(f"Analysis error: {e}")

# Additional information
st.sidebar.markdown("---")
st.sidebar.subheader("About This Demo")

st.sidebar.markdown("""
**Model**: BioBERT-based NER system
**Entity Types**: Diseases, Chemicals, Symptoms, Procedures
**Purpose**: Research and educational demonstration
""")

st.sidebar.markdown("""
**Privacy Features**:
- Optional de-identification
- PHI detection and warning
- No data storage
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Medical Entity Recognition Demo | Research Use Only | Not for Clinical Use</p>
</div>
""", unsafe_allow_html=True)
