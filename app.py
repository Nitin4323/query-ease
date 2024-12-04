import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from st_aggrid import AgGrid, GridOptionsBuilder,GridUpdateMode

from app import Backend

def sidebar():
    st.sidebar.title("DB Details")
    with st.sidebar.form(key='db_details'):
        host = st.text_input('Host', key="db_host")
        username = st.text_input('Username', key="db_user")
        password = st.text_input('Password', key="db_password", type='password')
        port = st.text_input('Port', key="db_port")
        database = st.text_input('Database', key="db_name")
        submit_button = st.form_submit_button(label='Submit')
        if submit_button:
            with st.spinner("generating meta data and vector storage ..."):
                Backend.initialize_app_orchestrator(host, port, database, username, password)
                st.success("Database details updated!")
                return True 
        
def show_dataframe (df,idx):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(filter='agTextColumnFilter' , sortable=True, editable=False)
    grid_options = gb.build()
    grid_options["domLayout"] = "autoHeight" 
    row_count = len(df)
    row_height = 25
    grid_height =min(max(row_count * row_height, 200), 400)  

    response = AgGrid(df,
                      gridOptions=grid_options,
                      height=grid_height,
                      width="100%",
                      enable_enterprise_modules=True,
                      update_mode=GridUpdateMode.MODEL_CHANGED)

    updated_df = pd.DataFrame(response["data"]) 
    st.download_button(label="Download",
                        data=Backend.get_excel_from_df(updated_df),
                        file_name="chat_data.xlsx",
                        mime="application/vnd.ms-excel",
                        icon='📥',
                        key=f"download_button_{idx}")

    st.markdown("""
                <style>
                .css-1e5imcs {padding: 0; margin: 0;}  /* Adjust padding/margin around container */
                .css-18e3th9 {padding: 0; margin: 0;}  /* Adjust padding/margin around grid */
                .ag-theme-alpine .ag-root-wrapper {overflow: hidden;} /* Hide overflow to prevent extra space */
                </style>
                """,
                unsafe_allow_html=True)
    analyse_graph = st.checkbox("Analysed using graphs...",key={idx})
    if analyse_graph :
        generate_visualizations(df=df)

def chat_interface():

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    with st.form(key='user_input_form', clear_on_submit=True):
        user_input = st.text_input("Ask a question:")
        submit = st.form_submit_button(label="Send")

    if submit and user_input:
        with st.spinner("Fetching llm response..."):
            response = Backend.execute_prompt(query=user_input, enable_prompt_layer=True)
        st.session_state['history'].append((user_input, response))

    for idx, (question, answer) in enumerate(reversed(st.session_state['history'])):
        st.write(f"**You**: {question}")
        if answer[0]:
            show_dataframe(answer[1].head(10),idx)
        else:
            st.write(f"**Bot**: {answer[1]}")

    st.write("<div id='bottom'></div>", unsafe_allow_html=True)
    st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)

def generate_visualizations(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) > 0:  
        for col in numeric_cols:
            st.subheader(f'Histogram for {col}')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Histogram for {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        if len(numeric_cols) > 1:
            st.subheader('Correlation Matrix')
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)

    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.subheader(f'Count Plot for {col}')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=df, x=col, ax=ax)
            ax.set_title(f'Count Plot for {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            st.pyplot(fig)

    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                st.subheader(f'Boxplot of {num_col} by {cat_col}')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
                ax.set_title(f'Boxplot of {num_col} by {cat_col}')
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
                st.pyplot(fig)

def main():
    st.title("QueryEase : Business Insights at Fingertips")
    sidebar()
    chat_interface()

if __name__ == "__main__":
    main()
