import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from sklearn.impute import KNNImputer
import io

# =============================================================================
# 1. CẤU HÌNH TRANG & CSS
# =============================================================================
st.set_page_config(page_title="Data Prep & Chat Dashboard", layout="wide")

# CSS custom cho giao diện và nút Chat nổi (Góc dưới bên phải)
st.markdown("""
<style>
    .floating-chat {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo Gemini (Giả lập)
GEMINI_API_KEY = ""
# Cấu hình API Key (Nên dùng file .env để bảo mật)
os.environ["GOOGLE_API_KEY"] = "GEMINI_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-flash')

# =============================================================================
# 2. KHỞI TẠO SESSION STATE (QUẢN LÝ PHIÊN LÀM VIỆC ĐỘC LẬP)
# =============================================================================
if 'dfs' not in st.session_state:
    st.session_state.dfs = {} # Lưu trữ các dataframe {tên_file: df}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# =============================================================================
# 3. HÀM HELPER CHO CHATBOT (LẤY THỐNG KÊ MÔ TẢ)
# =============================================================================
def get_desc_stats(table_name: str, columns: list = None) -> str:
    """Chatbot có thể gọi hàm này để lấy thống kê mô tả của bảng."""
    if table_name not in st.session_state.dfs:
        return f"Không tìm thấy bảng {table_name}"
    
    df = st.session_state.dfs[table_name]
    if columns:
        # Lọc các cột có tồn tại trong df
        valid_cols = [c for c in columns if c in df.columns]
        if not valid_cols: return "Các cột yêu cầu không tồn tại."
        df = df[valid_cols]
        
    stats = df.describe(include='all').to_string()
    return f"Thống kê mô tả của {table_name}:\n{stats}"

# =============================================================================
# 4. SIDEBAR - UPLOAD DATA
# =============================================================================
with st.sidebar:
    st.header("📂 Nhập Dữ Liệu")
    uploaded_files = st.file_uploader(
        "Tải lên tối đa 3 file CSV/Excel (Tối đa 30MB/file)", 
        type=['csv', 'xlsx'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.error("Chỉ được tải lên tối đa 3 file cùng lúc!")
        else:
            for file in uploaded_files:
                if file.size > 30 * 1024 * 1024:
                    st.error(f"File {file.name} vượt quá 30MB.")
                    continue
                
                # Lưu vào session_state nếu chưa có
                if file.name not in st.session_state.dfs:
                    try:
                        if file.name.endswith('.csv'):
                            df = pd.read_csv(file)
                        else:
                            df = pd.read_excel(file)
                        st.session_state.dfs[file.name] = df
                        st.success(f"Đã tải: {file.name}")
                    except Exception as e:
                        st.error(f"Lỗi đọc file {file.name}: {e}")

# =============================================================================
# 5. GIAO DIỆN CHÍNH (TABS)
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "🛠️ Xử lý dữ liệu", "📈 Visualize (Comming Soon)"])

# --- TAB 1: TỔNG QUAN ---
with tab1:
    if not st.session_state.dfs:
        st.info("Vui lòng tải dữ liệu ở thanh bên trái.")
    else:
        for file_name, df in st.session_state.dfs.items():
            st.subheader(f"📄 Bảng: {file_name}")
            col_data, col_stats = st.columns([3, 2])
            
            with col_data:
                # Giao diện Edit, Sort, Filter tích hợp sẵn của st.data_editor
                # Chiều cao ~250px tương đương khoảng 6 dòng dữ liệu
                edited_df = st.data_editor(df, height=250, use_container_width=True, key=f"editor_{file_name}")
                st.session_state.dfs[file_name] = edited_df # Cập nhật lại data nếu có edit
                
                # Ép kiểu Timeseries
                st.write("**Chuyển đổi Timeseries**")
                ts_col = st.selectbox("Chọn cột:", [""] + list(edited_df.columns), key=f"ts_col_{file_name}")
                if ts_col:
                    ts_type = st.selectbox("Định dạng:", ["datetime", "date", "time"], key=f"ts_type_{file_name}")
                    if st.button("Áp dụng ép kiểu", key=f"ts_btn_{file_name}"):
                        try:
                            if ts_type == "datetime":
                                st.session_state.dfs[file_name][ts_col] = pd.to_datetime(edited_df[ts_col])
                            elif ts_type == "date":
                                st.session_state.dfs[file_name][ts_col] = pd.to_datetime(edited_df[ts_col]).dt.date
                            elif ts_type == "time":
                                st.session_state.dfs[file_name][ts_col] = pd.to_datetime(edited_df[ts_col]).dt.time
                            st.success(f"Ép kiểu thành công cột {ts_col}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi: Cột này không phải định dạng {ts_type} hợp lệ. Chi tiết: {e}")

            with col_stats:
                st.write("**Thống kê mô tả**")
                cols_to_stat = st.multiselect("Chọn cột xem thống kê:", options=edited_df.columns, default=list(edited_df.columns), key=f"stat_cols_{file_name}")
                if cols_to_stat:
                    st.dataframe(edited_df[cols_to_stat].describe(include='all'), height=250, use_container_width=True)
            st.divider()

# --- TAB 2: XỬ LÝ DỮ LIỆU ---
with tab2:
    if st.session_state.dfs:
        selected_table = st.selectbox("Chọn bảng cần xử lý:", list(st.session_state.dfs.keys()))
        work_df = st.session_state.dfs[selected_table]
        
        c_stat, c_action = st.columns([1, 1])
        
        with c_stat:
            st.write("**Thống kê hiện tại**")
            st.dataframe(work_df.describe(), height=400)
            
        with c_action:
            target_col = st.selectbox("Chọn cột xử lý:", work_df.columns)
            action_type = st.radio("Hành động:", ["Xóa (Drop)", "Điền khuyết (Impute)"], horizontal=True)
            
            # Logic xác định điều kiện
            condition = st.selectbox("Dựa trên:", ["NaN (Dữ liệu trống)", "Giá trị IQR (Outliers)", "Giá trị X cụ thể", "Ngưỡng (>, <, >=, <=)"])
            
            val_X = None
            operator = None
            if condition == "Giá trị X cụ thể":
                val_X = st.text_input("Nhập X:")
            elif condition == "Ngưỡng (>, <, >=, <=)":
                col_op, col_val = st.columns(2)
                operator = col_op.selectbox("Toán tử", [">", "<", ">=", "<="])
                val_X = col_val.number_input("Nhập X (số):")

            impute_method = None
            if action_type == "Điền khuyết (Impute)":
                impute_method = st.selectbox("Phương pháp:", ["mean", "median", "mode", "Giá trị nhập tay", "LOCF (Forward fill)", "NOCB (Backward fill)", "Linear Interpolation", "KNN"])
                if impute_method == "Giá trị nhập tay":
                    manual_val = st.text_input("Nhập giá trị thay thế:")
            
            if st.button("🚀 Execute", type="primary"):
                try:
                    df_temp = work_df.copy()
                    
                    # Bước 1: Xác định Mask (các dòng cần xử lý) và gán thành NaN nếu là Impute
                    mask = pd.Series(False, index=df_temp.index)
                    if condition == "NaN (Dữ liệu trống)":
                        mask = df_temp[target_col].isna()
                    elif condition == "Giá trị X cụ thể" and val_X is not None:
                        mask = df_temp[target_col].astype(str) == str(val_X)
                    elif condition == "Ngưỡng (>, <, >=, <=)" and val_X is not None:
                        df_temp[target_col] = pd.to_numeric(df_temp[target_col], errors='coerce')
                        if operator == ">": mask = df_temp[target_col] > val_X
                        elif operator == "<": mask = df_temp[target_col] < val_X
                        elif operator == ">=": mask = df_temp[target_col] >= val_X
                        elif operator == "<=": mask = df_temp[target_col] <= val_X
                    elif condition == "Giá trị IQR (Outliers)":
                        Q1 = df_temp[target_col].quantile(0.25)
                        Q3 = df_temp[target_col].quantile(0.75)
                        IQR = Q3 - Q1
                        mask = (df_temp[target_col] < (Q1 - 1.5 * IQR)) | (df_temp[target_col] > (Q3 + 1.5 * IQR))

                    # Bước 2: Thực thi
                    if action_type == "Xóa (Drop)":
                        df_temp = df_temp[~mask]
                        st.success(f"Đã xóa {mask.sum()} dòng.")
                    
                    elif action_type == "Điền khuyết (Impute)":
                        # Đưa các giá trị cần xử lý về NaN trước
                        df_temp.loc[mask, target_col] = np.nan
                        
                        if impute_method == "mean": df_temp[target_col].fillna(df_temp[target_col].mean(), inplace=True)
                        elif impute_method == "median": df_temp[target_col].fillna(df_temp[target_col].median(), inplace=True)
                        elif impute_method == "mode": df_temp[target_col].fillna(df_temp[target_col].mode()[0], inplace=True)
                        elif impute_method == "Giá trị nhập tay": df_temp[target_col].fillna(manual_val, inplace=True)
                        elif impute_method == "LOCF (Forward fill)": df_temp[target_col].fillna(method='ffill', inplace=True)
                        elif impute_method == "NOCB (Backward fill)": df_temp[target_col].fillna(method='bfill', inplace=True)
                        elif impute_method == "Linear Interpolation": df_temp[target_col].interpolate(method='linear', inplace=True)
                        elif impute_method == "KNN":
                            # KNN yêu cầu dữ liệu số
                            imputer = KNNImputer(n_neighbors=5)
                            df_temp[target_col] = imputer.fit_transform(df_temp[[target_col]])
                        st.success("Impute thành công!")

                    # Cập nhật lại Session State
                    st.session_state.dfs[selected_table] = df_temp
                    st.rerun()

                except Exception as e:
                    st.error(f"Có lỗi xảy ra trong quá trình xử lý: {e}")

# --- TAB 3: VISUALIZE ---
with tab3:
    st.info("Khu vực phát triển biểu đồ Visualize (Sẽ cập nhật sau).")


# =============================================================================
# 6. CHATBOT GIAO DIỆN NỔI (Sử dụng st.popover)
# =============================================================================
st.markdown('<div class="floating-chat">', unsafe_allow_html=True)
with st.popover("💬 Trợ lý Gemini"):
    st.markdown("**Trợ lý phân tích dữ liệu**")
    
    # Hiển thị lịch sử
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])
        
    prompt = st.chat_input("Hỏi tôi về bộ dữ liệu của bạn...")
    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Mô phỏng phản hồi (Bạn sẽ tích hợp genai.generate_content tại đây)
        # Nếu prompt có nhắc đến "thống kê", bạn có thể gọi hàm get_desc_stats()
        response = f"Đây là phản hồi giả lập từ Gemini cho câu hỏi: '{prompt}'. Để bot tự động chuyển tab, bạn cần tích hợp Function Calling của Gemini SDK."
        
        st.chat_message("assistant").write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
st.markdown('</div>', unsafe_allow_html=True)