import streamlit as st

def apply_floating_bubble_style():
    st.markdown("""
    <style>
        /* 全局背景：深色漸層，營造神祕科技感 */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #E0E0E0;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }

        /* 隱藏 Streamlit 預設的 Header 和 Footer 以求美觀 */
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* 定義漂浮動畫 */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }

        /* 標題樣式 */
        .main-title {
            font-size: 3em;
            font-weight: 700;
            background: -webkit-linear-gradient(#00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 30px;
            text-shadow: 0 0 20px rgba(0, 198, 255, 0.3);
        }

        /* 通用氣泡卡片 (Glassmorphism) */
        .bubble-card {
            background: rgba(255, 255, 255, 0.05); /* 極淡的白色 */
            backdrop-filter: blur(12px);           /* 毛玻璃模糊 */
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            color: #ffffff;
            transition: all 0.3s ease;
        }

        /* 滑鼠懸停時的效果 */
        .bubble-card:hover {
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
            transform: scale(1.02);
        }

        /* 左側：科學藍色氣泡 */
        .sci-bubble {
            border-left: 5px solid #00B4D8;
            animation: float 6s ease-in-out infinite;
        }
        .sci-title {
            color: #00B4D8;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }

        /* 右側：風水金色氣泡 */
        .fs-bubble {
            border-left: 5px solid #FFD700;
            animation: float 6s ease-in-out infinite;
            animation-delay: 1s; /* 錯開動畫時間 */
        }
        .fs-title {
            color: #FFD700;
            font-weight: bold;
            font-size: 1.2em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }

        /* 底部：整合建議紫色氣泡 */
        .mediator-bubble {
            background: rgba(108, 92, 231, 0.1);
            border: 1px solid rgba(108, 92, 231, 0.3);
            border-radius: 25px;
            padding: 30px;
            margin-top: 20px;
        }
        .mediator-title {
            color: #a29bfe;
            font-size: 1.5em;
            text-align: center;
            margin-bottom: 15px;
        }

        /* 按鈕美化 */
        div.stButton > button {
            background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 10px 24px;
            font-weight: bold;
            transition: 0.3s;
            box-shadow: 0 0 15px rgba(0, 114, 255, 0.5);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 0 25px rgba(0, 114, 255, 0.8);
        }

    </style>
    """, unsafe_allow_html=True)
