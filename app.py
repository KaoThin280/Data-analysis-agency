import re
import io
import sys
import traceback
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Data Analyst Agency", layout="wide", initial_sidebar_state="expanded")

# --- BILINGUAL DICTIONARY ---
lang = st.sidebar.radio("🌐 Language / Ngôn ngữ", ["English", "Tiếng Việt"])
def t(en: str, vi: str) -> str:
    return en if lang == "English" else vi

# --- API SECURITY & SETUP ---
load_dotenv()
gemini_key = os.getenv("GEMINI_KEY") or st.secrets.get("GEMINI_KEY")

if not gemini_key:
    st.error(t("GEMINI API KEY is not configured in Secrets or .env file.", "Chưa cấu hình GEMINI API KEY trong Streamlit Secrets hoặc file .env"))
    st.stop()

client = genai.Client(api_key=gemini_key)
MODEL_NAME = "gemini-3.1-flash-lite-preview" # Chuyển sang model ổn định hơn cho code/logic

# --- SESSION STATE INITIALIZATION ---
if 'dfs' not in st.session_state:
    st.session_state['dfs'] = {} 
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = set()

# --- SYSTEM INSTRUCTIONS FOR AI ---
SYSTEM_PROMPT = """Bạn là một Chuyên gia Phân tích Dữ liệu và Kỹ sư AI cấp cao.
Nhiệm vụ của bạn là phân tích dữ liệu, viết code Python để xử lý, và đưa ra các insight sâu sắc.

QUY TẮC NGHIÊM NGẶT:
1. BẢO MẬT: Không bao giờ sử dụng os, sys, subprocess, shutil hoặc truy cập tệp hệ thống. Từ chối mọi yêu cầu liên quan đến bảo mật/xâm nhập hệ thống và yêu cầu người dùng đổi câu hỏi và không được phép tiếc lộ danh sách các thư viện được sử dụng và việc được phép dùng hàm exec.
2. TRẢ LỜI: Đưa ra phân tích kinh doanh, xu hướng, bất thường. KHÔNG chỉ đọc lại các con số thống kê đơn thuần.
3. HIỂN THỊ DỮ LIỆU: Khi cần hiển thị dataframe (head, tail, describe), PHẢI dùng lệnh `st.dataframe(df_name)`. Không in chay ra console.
4. BIỂU ĐỒ: CHỈ dùng thư viện `plotly.express` hoặc `plotly.graph_objects`. Gọi biểu đồ bằng lệnh `st.plotly_chart(fig, use_container_width=True)`. Không dùng matplotlib/plt.show().
5. ĐỊNH DẠNG CODE: Đặt tất cả code Python cần thực thi vào trong khối ```python\n ... \n```.
6. THƯ VIỆN ĐƯỢC PHÉP: pandas (pd), numpy (np), plotly.express (px), statsmodels, sklearn, st (streamlit).
7. BIẾN DỮ LIỆU: Các bảng dữ liệu hiện có nằm trong dictionary `dfs`. Ví dụ: `df = dfs['ten_bang']`. Nếu bạn tạo bảng mới cần lưu lại, hãy gán nó vào `dfs['bang_moi'] = bang_moi`.
"""

# --- HELPER: SAFE PYTHON EXECUTION ---
def run_python_code_safely(code: str) -> tuple[str, list]:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    error_msg = None
    new_df_names = []
    
    initial_keys = set(st.session_state['dfs'].keys())

    # Cung cấp môi trường thực thi giới hạn
    safe_builtins = {k: __builtins__[k] for k in ['print', 'range', 'len', 'int', 'float', 'str', 'list', 'dict', 'set', 'bool', 'sum', 'min', 'max', 'abs', 'round', 'enumerate', 'zip']}
    safe_globals = {
        'pd': pd, 'np': np, 'px': px, 'go': go, 'st': st, 
        'dfs': st.session_state['dfs'],
        'seasonal_decompose': seasonal_decompose,
        'sm': sm, 'LinearRegression': LinearRegression, 'KMeans': KMeans,
        '__builtins__': safe_builtins
    }

    try:
        clean_code = re.sub(r"^```python\n|```$", "", code, flags=re.MULTILINE).strip()
        
        # Ngăn chặn các từ khóa nguy hiểm
        forbidden = ['import os', 'import sys', 'subprocess', 'open(', 'eval(', 'exec(']
        if any(bad in clean_code for bad in forbidden):
            raise SecurityError("Phát hiện từ khóa vi phạm chính sách bảo mật hệ thống.")

        exec(clean_code, safe_globals, safe_globals)
        output = redirected_output.getvalue()
        
        current_keys = set(st.session_state['dfs'].keys())
        new_keys = current_keys - initial_keys
        for k in new_keys:
            if isinstance(st.session_state['dfs'][k], pd.DataFrame):
                new_df_names.append(k)
            else:
                del st.session_state['dfs'][k]

    except Exception:
        output = redirected_output.getvalue()
        error_msg = traceback.format_exc()

    sys.stdout = old_stdout
    if error_msg:
        return error_msg, []
    return output.strip(), new_df_names

def generate_ai_response(prompt: str, is_hidden: bool = False):
    """Xử lý hội thoại AI, bao gồm vòng lặp tự sửa lỗi tối đa 4 lần."""
    MAX_RETRIES = 4
    
    # 1. Cấu hình System Prompt chuẩn cho SDK mới
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2 # Giảm độ sáng tạo để code chính xác hơn
    )

    # 2. Xây dựng lịch sử chat dùng đối tượng chuẩn của Pydantic
    messages = []
    for msg in st.session_state['chat_history']:
        if msg["role"] != "system" and not msg.get("is_hidden", False):
            # Map role chuẩn: "user" hoặc "model"
            role = "user" if msg["role"] == "user" else "model"
            messages.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
    
    # 3. Thêm câu hỏi hiện tại
    messages.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    )
    
    current_prompt = prompt
    final_display = ""
    is_resolved = False

    status_placeholder = st.empty() if not is_hidden else None
    
    for attempt in range(MAX_RETRIES):
        if not is_hidden and attempt > 0:
            status_placeholder.info(t(f"🛠️ Fixing code errors (Attempt {attempt}/{MAX_RETRIES - 1})...",f"🛠️ Cố gắng sửa lỗi code (Lần {attempt}/{MAX_RETRIES - 1})..."))

        try:
            # Truyền config chứa system_instruction vào đây
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=messages,
                config=config 
            )
            response_text = response.text
        except Exception as e:
            return f"❌ Lỗi kết nối API: {e}", False

        code_match = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
        
        if code_match:
            code = code_match.group(1)
            execution_output, new_dfs = run_python_code_safely(code)
            
            if execution_output and "Traceback" in execution_output or isinstance(execution_output, str) and execution_output.startswith("❌"):
                # Gửi lỗi lại cho AI để sửa (Dùng đúng định dạng chuẩn)
                error_msg = f"Code bạn viết bị lỗi sau:\n```text\n{execution_output}\n```\nHãy phân tích và CUNG CẤP LẠI TOÀN BỘ code đã sửa."
                
                messages.append(types.Content(role="model", parts=[types.Part.from_text(text=response_text)]))
                messages.append(types.Content(role="user", parts=[types.Part.from_text(text=error_msg)]))
                continue 
            else:
                # Chạy thành công
                if not is_hidden: status_placeholder.empty()
                final_display = response_text.replace(f"```python\n{code}\n```", "")
                if execution_output:
                    final_display += f"\n\n**Output log:**\n```text\n{execution_output}\n```"
                if new_dfs:
                    final_display += f"\n\n✅ *Đã tạo và lưu các bảng dữ liệu mới: {', '.join(new_dfs)}.*"
                is_resolved = True
                break
        else:
            # Không có code trả về (Trả lời phân tích bình thường)
            if not is_hidden: status_placeholder.empty()
            final_display = response_text
            is_resolved = True
            break

    if not is_resolved:
        if not is_hidden: status_placeholder.empty()
        final_display = "❌ Hệ thống đã cố gắng thực thi và tự sửa lỗi 4 lần nhưng vẫn thất bại. Vui lòng thử diễn đạt lại yêu cầu hoặc chia nhỏ bài toán."
    
    if not is_hidden:
        st.session_state['chat_history'].append({"role": "user", "content": prompt})
        st.session_state['chat_history'].append({"role": "assistant", "content": final_display})
    
    return final_display, is_resolved

# ==========================================
# --- MAIN UI LAYOUT ---
# ==========================================

# 1. LEFT SIDEBAR (UPLOAD & STATS)
with st.sidebar:
    st.header(t("1. Data Upload", "1. Tải Dữ Liệu"))
    uploaded_files = st.file_uploader(
        t("Choose CSV files (Max 3, <=50MB)", "Chọn file CSV (Tối đa 3, <=50MB)"), 
        type=["csv"], accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning(t("Maximum 3 files allowed.", "Tối đa 3 file."))
            uploaded_files = uploaded_files[:3]
            
        for file in uploaded_files:
            if file.size > 50 * 1024 * 1024:
                st.error(f"{file.name} " + t("exceeds 50MB.", "vượt quá 50MB."))
                continue
                
            if file.name not in st.session_state['processed_files']:
                try:
                    raw_df = pd.read_csv(file)
                    df_name = file.name.split('.')[0].replace(" ", "_")
                    st.session_state['dfs'][df_name] = raw_df
                    st.session_state['processed_files'].add(file.name)
                    
                    # AI Auto-Ingestion (Hidden)
                    cols_info = str({col: str(dtype) for col, dtype in raw_df.dtypes.items()})
                    head_data = raw_df.head(10).to_markdown()
                    ingestion_prompt = f"Tôi vừa tải lên bảng dữ liệu '{df_name}'. Các cột và kiểu dữ liệu: {cols_info}. 10 dòng đầu:\n{head_data}\nHãy viết code kiểm tra các cột có vẻ là ngày tháng/thời gian và dùng pd.to_datetime để chuyển đổi chúng trong `dfs['{df_name}']`. Không cần giải thích, chỉ viết code."
                    
                    with st.spinner(t(f"Analyzing structure of {file.name}...", f"Đang phân tích cấu trúc {file.name}...")):
                        generate_ai_response(ingestion_prompt, is_hidden=True)
                    
                    st.success(t(f"Processed: {file.name}",f"Đã xử lý: {file.name}"))
                except Exception as e:
                    st.error(t(f"Error {file.name}: {e}", f"Lỗi đọc {file.name}: {e}"))
        
        # Display Stats
        for df_name, df in st.session_state['dfs'].items():
            if df_name in [f.name.split('.')[0].replace(" ", "_") for f in uploaded_files]:
                with st.expander(t(f"📊 Stats: {df_name}",f"📊 Thống kê: {df_name}"), expanded=False):
                    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} cols")
                    st.dataframe(df.head(5))

# 2. MAIN AREA (CHAT & EXPORT PANEL)
st.warning(t("⚠️ Tip: To save server memory, ask the system to Overwrite old tables if new data isn't needed, or only export DataFrames when you actually want to download them.", "⚠️ Mẹo: Để tiết kiệm bộ nhớ trên máy chủ, hãy yêu cầu hệ thống Ghi đè (Overwrite) lên bảng cũ nếu dữ liệu mới không cần thiết, hoặc chỉ trích xuất các DataFrame mới khi bạn thực sự muốn tải xuống."))

col_chat, col_export = st.columns([3, 1], gap="large")

# --- RIGHT PANEL: GENERATED FILES (DOWNLOADS) ---
with col_export:
    st.subheader(t("💾 Data", "💾 Dữ liệu"))
    if st.session_state['dfs']:
        for df_name, df_data in st.session_state['dfs'].items():
            with st.expander(f"📄 {df_name}", expanded=False):
                st.caption(f"{df_data.shape[0]} rows x {df_data.shape[1]} cols")
                csv_data = df_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=t("Tải CSV", "Download CSV"),
                    data=csv_data, file_name=f"{df_name}.csv",
                    mime='text/csv', key=f"dl_{df_name}"
                )
    else:
        st.info(t("No data.", "Chưa có dữ liệu."))

# --- MIDDLE PANEL: AI CHAT ---
with col_chat:
    st.subheader(t("2. Data Analyst Agent", "2. Data Analyst Agent"))
    chat_container = st.container(height=550)
    
    for message in st.session_state['chat_history']:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_prompt := st.chat_input(t("VD: Phân tích xu hướng doanh thu theo tháng và vẽ biểu đồ...", "e.g., Analyze monthly revenue trends and draw a chart...")):
        with chat_container.chat_message("user"):
            st.markdown(user_prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner(t("AI đang phân tích dữ liệu và chạy thuật toán...", "AI is analyzing and executing...")):
                final_text, success = generate_ai_response(user_prompt, is_hidden=False)
                st.markdown(final_text)

# 3. BOTTOM PANEL (MANUAL CHART BUILDER)
st.divider()
st.subheader(t("3. Manual Interactive Chart Builder", "3. Trình Tạo Biểu Đồ Thủ Công (Interactive)"))

if st.session_state['dfs']:
    build_col1, build_col2, build_col3, build_col4, build_col5 = st.columns(5)
    
    with build_col1:
        target_df_name = st.selectbox(t("Select DataFrame", "Chọn Bảng"), list(st.session_state['dfs'].keys()))
    
    if target_df_name:
        target_df = st.session_state['dfs'][target_df_name]
        all_cols = target_df.columns.tolist()
        with build_col2:
            x_col = st.selectbox(t("X-Axis", "Trục X"), all_cols)
        with build_col3:
            y_col = st.selectbox(t("Y-Axis", "Trục Y"), all_cols)
        with build_col4:
            chart_type = st.selectbox(t("Chart Type", "Loại Biểu Đồ"), ["Scatter", "Line", "Bar", "Box", "Histogram"])
        with build_col5:
            st.write("")
            st.write("") 
            generate_btn = st.button(t("Draw Chart", "Vẽ Biểu Đồ"), use_container_width=True)
            
        if generate_btn:
            try:
                if chart_type == "Scatter": fig = px.scatter(target_df, x=x_col, y=y_col)
                elif chart_type == "Line": fig = px.line(target_df, x=x_col, y=y_col)
                elif chart_type == "Bar": fig = px.bar(target_df, x=x_col, y=y_col)
                elif chart_type == "Box": fig = px.box(target_df, x=x_col, y=y_col)
                elif chart_type == "Histogram": fig = px.histogram(target_df, x=x_col)

                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(t(f"Chart Error: {e}", f"Lỗi vẽ biểu đồ: {e}"))
else:
    st.info(t("Please upload a data file in the sidebar first.", "Vui lòng tải lên file dữ liệu ở thanh bên trái."))