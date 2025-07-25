import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import io
import json

# Page configuration
st.set_page_config(
    page_title="Text Classification Tool",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for dictionaries
if 'dictionaries' not in st.session_state:
    st.session_state.dictionaries = {
        'urgency_marketing': [
            'limited', 'limited time', 'limited run', 'limited edition', 'order now',
            'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
            'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
            'expires soon', 'final hours', 'almost gone'
        ],
        'exclusive_marketing': [
            'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
            'members only', 'vip', 'special access', 'invitation only',
            'premium', 'privileged', 'limited access', 'select customers',
            'insider', 'private sale', 'early access'
        ]
    }

def classify_text(text, dictionaries):
    """Classify text based on dictionary matches"""
    if pd.isna(text):
        return {}
    
    text_lower = str(text).lower()
    results = defaultdict(list)
    
    for category, terms in dictionaries.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term.lower()) + r'\b', text_lower):
                results[category].append(term)
    
    return dict(results)

def process_dataframe(df, text_column, dictionaries):
    """Process DataFrame and add classification results"""
    df = df.copy()
    
    # Apply classification
    df['classifications'] = df[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Create separate columns for each category
    for category in dictionaries.keys():
        df[f'{category}_matches'] = df['classifications'].apply(
            lambda x: ', '.join(x.get(category, []))
        )
        df[f'{category}_count'] = df['classifications'].apply(
            lambda x: len(x.get(category, []))
        )
    
    return df

def create_summary_stats(df, dictionaries):
    """Create summary statistics"""
    summary = {}
    for category in dictionaries.keys():
        total_matches = df[f'{category}_count'].sum()
        rows_with_matches = (df[f'{category}_count'] > 0).sum()
        summary[category] = {
            'total_matches': total_matches,
            'rows_with_matches': rows_with_matches,
            'percentage': (rows_with_matches / len(df) * 100) if len(df) > 0 else 0
        }
    return summary

# App title and description
st.title("📊 Text Classification Tool")
st.markdown("""
This tool classifies text based on customizable dictionaries. Upload your CSV file and modify the classification categories as needed.
""")

# Sidebar for dictionary management
st.sidebar.header("📚 Dictionary Management")

# Dictionary editor
st.sidebar.subheader("Edit Categories")
categories_to_remove = []

for category in list(st.session_state.dictionaries.keys()):
    with st.sidebar.expander(f"📝 {category.replace('_', ' ').title()}"):
        # Display current terms
        current_terms = st.session_state.dictionaries[category]
        terms_text = '\n'.join(current_terms)
        
        # Text area for editing terms
        new_terms = st.text_area(
            f"Terms (one per line):",
            value=terms_text,
            height=100,
            key=f"terms_{category}"
        )
        
        # Update terms
        if st.button(f"Update {category}", key=f"update_{category}"):
            st.session_state.dictionaries[category] = [
                term.strip() for term in new_terms.split('\n') if term.strip()
            ]
            st.success(f"Updated {category}")
        
        # Remove category
        if st.button(f"Remove {category}", key=f"remove_{category}"):
            categories_to_remove.append(category)

# Remove categories
for category in categories_to_remove:
    del st.session_state.dictionaries[category]
    st.sidebar.success(f"Removed {category}")

# Add new category
st.sidebar.subheader("➕ Add New Category")
with st.sidebar.form("add_category_form"):
    new_category_name = st.text_input("Category Name:")
    new_category_terms = st.text_area("Terms (one per line):")
    
    if st.form_submit_button("Add Category"):
        if new_category_name and new_category_terms:
            category_key = new_category_name.lower().replace(' ', '_')
            terms_list = [term.strip() for term in new_category_terms.split('\n') if term.strip()]
            st.session_state.dictionaries[category_key] = terms_list
            st.sidebar.success(f"Added category: {new_category_name}")
        else:
            st.sidebar.error("Please provide both category name and terms")

# Export/Import dictionaries
st.sidebar.subheader("💾 Dictionary Import/Export")

# Export dictionary
if st.sidebar.button("Export Dictionary"):
    dict_json = json.dumps(st.session_state.dictionaries, indent=2)
    st.sidebar.download_button(
        label="Download Dictionary JSON",
        data=dict_json,
        file_name="classification_dictionary.json",
        mime="application/json"
    )

# Import dictionary
uploaded_dict = st.sidebar.file_uploader("Import Dictionary", type=['json'])
if uploaded_dict is not None:
    try:
        imported_dict = json.load(uploaded_dict)
        st.session_state.dictionaries = imported_dict
        st.sidebar.success("Dictionary imported successfully!")
    except Exception as e:
        st.sidebar.error(f"Error importing dictionary: {e}")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📄 File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Column selection
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            if text_columns:
                selected_column = st.selectbox(
                    "Select the text column to classify:",
                    text_columns,
                    index=0 if 'Statement' in text_columns else 0
                )
                
                if st.button("🔍 Classify Text", type="primary"):
                    with st.spinner("Processing..."):
                        # Process the data
                        df_classified = process_dataframe(df, selected_column, st.session_state.dictionaries)
                        
                        # Store in session state
                        st.session_state.classified_df = df_classified
                        st.session_state.text_column = selected_column
                        
                        st.success("Classification completed!")
            else:
                st.error("No text columns found in the uploaded file.")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

with col2:
    st.header("📊 Current Dictionary")
    
    if st.session_state.dictionaries:
        for category, terms in st.session_state.dictionaries.items():
            with st.expander(f"{category.replace('_', ' ').title()} ({len(terms)} terms)"):
                st.write(", ".join(terms[:10]))
                if len(terms) > 10:
                    st.write(f"... and {len(terms) - 10} more terms")
    else:
        st.info("No categories defined yet.")

# Results section
if 'classified_df' in st.session_state:
    st.header("📊 Classification Results")
    
    df_classified = st.session_state.classified_df
    text_column = st.session_state.text_column
    
    # Summary statistics
    summary = create_summary_stats(df_classified, st.session_state.dictionaries)
    
    st.subheader("📈 Summary Statistics")
    cols = st.columns(len(summary))
    
    for i, (category, stats) in enumerate(summary.items()):
        with cols[i]:
            st.metric(
                label=category.replace('_', ' ').title(),
                value=f"{stats['rows_with_matches']}/{len(df_classified)}",
                delta=f"{stats['percentage']:.1f}%"
            )
            st.caption(f"Total matches: {stats['total_matches']}")
    
    # Detailed results
    st.subheader("🔍 Detailed Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        filter_category = st.selectbox(
            "Filter by category:",
            ["All"] + list(st.session_state.dictionaries.keys()),
            format_func=lambda x: x.replace('_', ' ').title() if x != "All" else x
        )
    
    with col2:
        show_only_matches = st.checkbox("Show only rows with matches", value=False)
    
    # Apply filters
    display_df = df_classified.copy()
    
    if filter_category != "All":
        if show_only_matches:
            display_df = display_df[display_df[f'{filter_category}_count'] > 0]
    elif show_only_matches:
        # Show rows with any matches
        match_columns = [f'{cat}_count' for cat in st.session_state.dictionaries.keys()]
        display_df = display_df[display_df[match_columns].sum(axis=1) > 0]
    
    # Select columns to display
    base_columns = list(df_classified.columns[:df_classified.columns.get_loc('classifications')])
    match_columns = [f'{cat}_matches' for cat in st.session_state.dictionaries.keys()]
    count_columns = [f'{cat}_count' for cat in st.session_state.dictionaries.keys()]
    
    display_columns = base_columns + match_columns + count_columns
    
    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        height=400
    )
    
    # Download results
    st.subheader("💾 Download Results")
    
    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df_classified.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📥 Download Classified Data (CSV)",
        data=csv_data,
        file_name="classified_results.csv",
        mime="text/csv"
    )
    
    # Show sample of detailed results
    st.subheader("📝 Sample Detailed Results")
    
    sample_size = min(5, len(display_df))
    if sample_size > 0:
        for i, (_, row) in enumerate(display_df.head(sample_size).iterrows()):
            with st.expander(f"Row {i+1}: {str(row[text_column])[:100]}..."):
                st.write(f"**Full text:** {row[text_column]}")
                
                for category in st.session_state.dictionaries.keys():
                    matches = row[f'{category}_matches']
                    if matches:
                        st.write(f"**{category.replace('_', ' ').title()}:** {matches}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar to customize your classification categories and terms.")
