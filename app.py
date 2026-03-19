import re
import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import traceback
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Data Analyst Agency", layout="wide", initial_sidebar_state="expanded")

# --- BILINGUAL DICTIONARY ---
lang = st.sidebar.radio("🌐 Language / Ngôn ngữ", ["English", "Tiếng Việt"])
def t(en: str, vi: str) -> str:
    return en if lang == "English" else vi

# --- API SECURITY & SETUP ---
load_dotenv()
gemini_key = os.getenv("GEMINI_KEY")
if not gemini_key:
    st.error(t("GEMINI KEY is not configured in the .env file.", "Chưa cấu hình GEMINI API KEY trong file .env"))
    st.stop()

genai.configure(api_key=gemini_key)
# Using the requested model name. If the API rejects it, update to 'gemini-1.5-flash' or 'gemini-2.0-flash'
MODEL_NAME = "gemini-3.1-flash-lite-preview" 
model = genai.GenerativeModel(MODEL_NAME)

# --- SESSION STATE INITIALIZATION ---
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {} # Stores ALL dataframes (inputs + generated)
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = set()

# --- HELPER: LLM DATETIME CONVERTER ---
def identify_and_convert_datetime(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    columns_info = {col: str(dtype) for col, dtype in df.dtypes.items() if dtype == 'object'}
    if not columns_info:
        return df
    
    prompt = f"""
    Analyze these column names and their string types from a dataset: {columns_info}.
    Return a comma-separated list of column names that likely contain dates or timestamps.
    If none, return exactly "NONE". Do not explain, just return the list.
    """
    try:
        response = model.generate_content(prompt)
        result = response.text.strip().replace('`', '').strip()
        if result.upper() != "NONE":
            dt_cols = [c.strip() for c in result.split(',')]
            for col in dt_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
                    except Exception:
                        pass
    except Exception as e:
        print(f"Datetime LLM Error: {e}")
    return df

# --- HELPER: SAFE PYTHON EXECUTION ---
def run_python_code_safely(code: str) -> tuple[str, list]:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    error_msg = None
    new_df_names = []
    
    # Track existing keys before execution
    initial_keys = set(st.session_state['dfs'].keys())

    safe_globals = {
        'pd': pd,
        'np': np,
        'px': px,
        'st': st,
        'dfs': st.session_state['dfs'], # Pass the dictionary of all dataframes
        '__builtins__': __builtins__
    }

    try:
        # Pre-process code to remove markdown wrappers if AI included them
        clean_code = re.sub(r"^```python\n|```$", "", code, flags=re.MULTILINE).strip()
        exec(clean_code, safe_globals, {})
        output = redirected_output.getvalue()
        
        # Check if AI created new dataframes in the 'dfs' dictionary
        current_keys = set(st.session_state['dfs'].keys())
        new_keys = current_keys - initial_keys
        
        for k in new_keys:
            if isinstance(st.session_state['dfs'][k], pd.DataFrame):
                new_df_names.append(k)
            else:
                # Remove if they injected non-dataframes
                del st.session_state['dfs'][k]

    except Exception:
        output = redirected_output.getvalue()
        error_msg = f"❌ Execution Error:\n{traceback.format_exc()}"

    sys.stdout = old_stdout
    if error_msg:
        return error_msg, []
    return output.strip(), new_df_names

# ==========================================
# --- MAIN UI LAYOUT ---
# ==========================================

# 1. LEFT SIDEBAR (UPLOAD & STATS)
with st.sidebar:
    st.header(t("1. Data Upload", "1. Tải Dữ Liệu"))
    uploaded_files = st.file_uploader(
        t("Choose CSV files (Max 3, <=50MB)", "Chọn file CSV (Tối đa 3 file, <=50MB)"), 
        type=["csv"], accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning(t("Please upload a maximum of 3 files.", "Vui lòng tải lên tối đa 3 file."))
            uploaded_files = uploaded_files[:3]
            
        for file in uploaded_files:
            if file.size > 50 * 1024 * 1024:
                st.error(f"{file.name} " + t("exceeds 50MB limit.", "vượt quá giới hạn 50MB."))
                continue
                
            if file.name not in st.session_state['processed_files']:
                try:
                    raw_df = pd.read_csv(file)
                    # Use Gemini to auto-convert datetimes
                    processed_df = identify_and_convert_datetime(raw_df, file.name)
                    df_name = file.name.split('.')[0].replace(" ", "_")
                    
                    st.session_state['dfs'][df_name] = processed_df
                    st.session_state['processed_files'].add(file.name)
                    st.success(t(f"Processed: {file.name}", f"Đã xử lý: {file.name}"))
                    
                except Exception as e:
                    st.error(t(f"Error reading {file.name}: {e}", f"Lỗi đọc {file.name}: {e}"))
        
        # Display Stats for all uploaded DataFrames
        for df_name, df in st.session_state['dfs'].items():
            if df_name in [f.name.split('.')[0].replace(" ", "_") for f in uploaded_files]:
                with st.expander(t(f"📊 Stats: {df_name}", f"📊 Thống kê: {df_name}"), expanded=False):
                    stats_df = pd.DataFrame(index=df.columns)
                    stats_df['Type'] = df.dtypes
                    stats_df['Count'] = df.count()
                    stats_df['NaN_Count'] = df.isna().sum()
                    stats_df['Unique'] = df.nunique()
                    
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    for ct in ['Mean', 'Min', '25% (Q1)', '50% (Med)', '75% (Q3)', 'Max']:
                        stats_df[ct] = np.nan
                        
                    if len(numeric_cols) > 0:
                        desc = df[numeric_cols].describe().T
                        stats_df.update(desc.rename(columns={'mean':'Mean', 'min':'Min', '25%':'25% (Q1)', '50%':'50% (Med)', '75%':'75% (Q3)', 'max':'Max'}))
                    
                    st.dataframe(stats_df.round(2))

# 2. MAIN AREA (CHAT, MANUAL CHARTS, & EXPORT PANEL)
col_chat, col_export = st.columns([3, 1], gap="large")

# --- RIGHT PANEL: GENERATED FILES (DOWNLOADS) ---
with col_export:
    st.subheader(t("💾 Processed Data", "💾 Dữ Liệu Đã Xử Lý"))
    st.markdown(t("DataFrames generated by AI will appear here.", "Các DataFrame do AI tạo ra sẽ hiển thị ở đây."))
    
    if st.session_state['dfs']:
        for df_name, df_data in st.session_state['dfs'].items():
            with st.expander(f"📄 {df_name}", expanded=True):
                st.caption(f"{df_data.shape[0]} rows x {df_data.shape[1]} cols")
                csv_data = df_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=t("Download CSV", "Tải CSV"),
                    data=csv_data,
                    file_name=f"{df_name}.csv",
                    mime='text/csv',
                    key=f"dl_{df_name}"
                )
    else:
        st.info(t("No data available yet.", "Chưa có dữ liệu."))

# --- MIDDLE PANEL: AI CHAT ---
with col_chat:
    st.subheader(t("2. Chat With Data Agent", "2. Chat Với Data Agent"))
    chat_container = st.container(height=500)
    
    for message in st.session_state['chat_history']:
        if message["role"] != "system": # Hide system context messages from UI
            with chat_container.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_prompt := st.chat_input(t("e.g., 'Filter sales > 1000 and save as high_sales'", "VD: 'Lọc doanh thu > 1000 và lưu thành bảng high_sales'")):
        
        st.session_state['chat_history'].append({"role": "user", "content": user_prompt})
        with chat_container.chat_message("user"):
            st.markdown(user_prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner(t("AI is analyzing...", "AI đang phân tích...")):
                # Construct System Prompt with current schema
                df_schemas = []
                for name, d in st.session_state['dfs'].items():
                    cols = ", ".join([f"'{c}' ({dt})" for c, dt in zip(d.columns, d.dtypes)])
                    df_schemas.append(f"DataFrame '{name}': {cols}")
                schema_context = "\n".join(df_schemas)
                
                system_instruction = f"""
                You are an expert Data Analyst AI. You have access to a dictionary of DataFrames named `dfs`.
                Current available dataframes:
                {schema_context}
                
                INSTRUCTIONS:
                - To analyze data, create charts, or modify data, output Python code wrapped in ```python ... ```.
                - NEVER use pd.read_csv. Only use data inside the `dfs` dictionary (e.g., `df = dfs['my_data']`).
                - To create a NEW dataframe for the user to download (e.g. filtering, imputing, aggregating), simply assign it to a new key in `dfs`. Example: `dfs['new_table'] = dfs['old_table'].dropna()`. Do NOT reassign the `dfs` variable itself.
                - To plot, use `plotly.express` (imported as `px`) and render with `st.plotly_chart(fig)`.
                - Print any direct text answers inside the python code using `print()`.
                """

                # Format history for Gemini API
                formatted_history = []
                for msg in st.session_state['chat_history'][:-1]: # exclude the latest prompt
                    role = "user" if msg["role"] == "user" else "model"
                    formatted_history.append({"role": role, "parts": [msg["content"]]})

                chat_session = model.start_chat(history=formatted_history)
                response = chat_session.send_message(system_instruction + "\n\nUser request: " + user_prompt)
                
                response_text = response.text
                
                # Check if AI generated code
                code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
                
                if code_match:
                    code = code_match.group(1)
                    with st.expander(t("🛠️ View AI Python Code", "🛠️ Xem mã Python AI sinh ra"), expanded=False):
                        st.code(code, language="python")
                    
                    with st.spinner(t("🔄 Executing code...", "🔄 Đang chạy code...")):
                        execution_output, new_dfs = run_python_code_safely(code)
                    
                    if execution_output:
                        if execution_output.startswith("❌"):
                            st.error(execution_output)
                            final_display = execution_output
                        else:
                            st.write(t("**📊 Execution Output:**", "**📊 Kết quả chạy code:**"))
                            st.code(execution_output, language="text")
                            final_display = f"{response_text}\n\n**Output:**\n```text\n{execution_output}\n```"
                    else:
                        final_display = response_text
                        st.markdown(response_text.replace(f"```python\n{code}\n```", ""))
                    
                    # If new dataframes were created, feed that info back to the model's memory
                    if new_dfs:
                        new_schemas = []
                        for n in new_dfs:
                            cols = ", ".join(st.session_state['dfs'][n].columns)
                            new_schemas.append(f"'{n}' with columns: {cols}")
                        sys_msg = f"System Update: Successfully created and saved new dataframes: {', '.join(new_schemas)}. The user can now see and download them."
                        st.session_state['chat_history'].append({"role": "system", "content": sys_msg})
                        st.success(t(f"Successfully generated new tables: {', '.join(new_dfs)}", f"Đã tạo bảng mới thành công: {', '.join(new_dfs)}"))
                else:
                    final_display = response_text
                    st.markdown(final_display)

                st.session_state['chat_history'].append({"role": "assistant", "content": final_display})

# 3. BOTTOM PANEL (MANUAL CHART BUILDER)
st.divider()
st.subheader(t("3. Manual Chart Builder", "3. Trình Tạo Biểu Đồ Thủ Công"))
st.markdown(t("Select any available dataframe below to instantly generate interactive charts.", "Chọn bất kỳ bảng dữ liệu nào dưới đây để tạo biểu đồ tương tác."))

if st.session_state['dfs']:
    build_col1, build_col2, build_col3, build_col4, build_col5 = st.columns(5)
    
    with build_col1:
        target_df_name = st.selectbox(t("Select DataFrame", "Chọn Bảng"), list(st.session_state['dfs'].keys()))
    
    if target_df_name:
        target_df = st.session_state['dfs'][target_df_name]
        all_cols = target_df.columns.tolist()
        num_cols = target_df.select_dtypes(include=['number']).columns.tolist()
        
        with build_col2:
            x_col = st.selectbox(t("X-Axis", "Trục X"), all_cols)
        with build_col3:
            y_col = st.selectbox(t("Y-Axis", "Trục Y"), all_cols)
        with build_col4:
            chart_type = st.selectbox(t("Chart Type", "Loại Biểu Đồ"), ["Scatter", "Line", "Bar", "Box", "Histogram"])
        with build_col5:
            st.write("")
            st.write("") # Spacing alignment
            generate_btn = st.button(t("Complete & Draw", "Hoàn tất & Vẽ"), use_container_width=True)
            
        if generate_btn:
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(target_df, x=x_col, y=y_col, title=f"Scatter: {y_col} vs {x_col}")
                elif chart_type == "Line":
                    fig = px.line(target_df, x=x_col, y=y_col, title=f"Line: {y_col} over {x_col}")
                elif chart_type == "Bar":
                    fig = px.bar(target_df, x=x_col, y=y_col, title=f"Bar: {y_col} by {x_col}")
                elif chart_type == "Box":
                    fig = px.box(target_df, x=x_col, y=y_col, title=f"Boxplot: {y_col} grouped by {x_col}")
                elif chart_type == "Histogram":
                    fig = px.histogram(target_df, x=x_col, title=f"Histogram: Distribution of {x_col}")
                
                # Render using Streamlit's Plotly container for built-in interactivity
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(t(f"Could not generate chart: {e}", f"Không thể tạo biểu đồ: {e}"))
else:
    st.info(t("Upload a file first to use the Chart Builder.", "Tải lên một file để sử dụng Trình Tạo Biểu Đồ."))
