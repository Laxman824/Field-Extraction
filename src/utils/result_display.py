def display_batch_results(results):
    """Display results for multiple documents"""
    st.header("Processing Results")
    
    # Create tabs for different views
    tabs = st.tabs(["Summary", "Details", "Download"])
    
    with tabs[0]:
        display_summary(results)
    
    with tabs[1]:
        display_details(results)
    
    with tabs[2]:
        provide_download_options(results)