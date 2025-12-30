import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import styles
from logic import LogicCore
import time

st.set_page_config(
    page_title="HarmonyChainSpace", 
    layout="wide", 
    page_icon="☯️"
)
styles.apply_floating_bubble_style()

# 初始化核心
if "core" not in st.session_state:
    st.session_state.core = LogicCore()

core = st.session_state.core

# 初始化 Session State
if "arch_result" not in st.session_state: st.session_state.arch_result = None
if "fs_result" not in st.session_state: st.session_state.fs_result = None
if "mediator_result" not in st.session_state: st.session_state.mediator_result = None
if "last_request" not in st.session_state: st.session_state.last_request = 0
if "prompt_content" not in st.session_state: st.session_state.prompt_content = ""


st.markdown('<div class="main-title">☯️ HarmonyChainSpace · 雙軌系統 </div>', unsafe_allow_html=True)

# ==========================================
# 🎛️ 側邊欄：設定與 API Key
# ==========================================
with st.sidebar:
    st.header("🔑 模型與工具設定")
    
    # 1. 選擇供應商
    provider = st.selectbox("1. 選擇推理大腦", ["Groq", "OpenAI", "Gemini"])
    
    # 2. 根據選擇顯示對應的 Key 輸入框與 2025 最新模型
    api_key = ""
    model_name = ""
    
    if provider == "Groq":
        api_key = st.text_input("Groq API Key", type="password", help="推薦 Llama 3.3")
        model_name = st.selectbox("模型版本", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    
    elif provider == "OpenAI":
        api_key = st.text_input("OpenAI API Key", type="password", help="推薦 GPT-4o")
        model_name = st.selectbox("模型版本", ["gpt-4o", "gpt-4o-mini"])
    
    elif provider == "Gemini":
        api_key = st.text_input("Google AI Studio Key", type="password", help="推薦 Gemini 2.5 Flash")
        model_name = st.selectbox("模型版本", ["gemini-2.5-flash", "gemini-1.5-pro"])

    st.markdown("---")

    # 3. 視覺與繪圖設定
    st.caption("🎨 視覺與繪圖工具")
    
    gemini_vision_key = st.text_input("Gemini Vision Key (選填)", type="password", help="若需分析圖片，請填入 Google Key。若上方已選 Gemini 則可留空。")
    if provider == "Gemini" and api_key and not gemini_vision_key:
        gemini_vision_key = api_key

    paint_mode = st.radio(
        "繪圖引擎", 
        ["Pollinations (免費/無限)", "Hugging Face (需Token)", "關閉繪圖"],
        help="Pollinations 使用 Flux 模型且完全免費"
    )
    
    hf_token = ""
    if paint_mode == "Hugging Face (需Token)":
        hf_token = st.text_input("Hugging Face Token", type="password")

    st.markdown("---")
    
    if st.button("🔄 重置系統狀態"):
        for key in ["arch_result", "fs_result", "mediator_result", "prompt_content"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

    uploaded_file = st.file_uploader("上傳空間實景", type=["jpg", "png", "jpeg"])
    image_desc = ""
    if uploaded_file:
        st.image(uploaded_file, caption="原始空間", use_container_width=True)
        if st.button("👁️ 分析圖片"):
            if not gemini_vision_key:
                st.error("❌ 請輸入 Gemini Vision Key 才能進行視覺分析")
            else:
                with st.spinner("Gemini 3.0 Pro 正在觀察圖片細節..."):
                    image_bytes = uploaded_file.getvalue()
                    image_desc = core.analyze_image(image_bytes, gemini_vision_key, uploaded_file.type)
                    st.success("視覺分析完成")
                    with st.expander("筆記"): st.write(image_desc)

    with st.expander("💡 靈感圖書館 (點擊自動帶入)"):
        st.caption("點擊下方按鈕，自動填寫經典的「科學 vs 風水」衝突場景：")
        scenarios = {
            "📏 樓梯法規陷阱\n\n(測試第33/36條)": "我想在客廳做一個旋轉樓梯，為了省空間，樓梯寬度只想做 70 公分。另外為了極簡美感，我不打算裝扶手，或者只裝 50 公分高的隱形玻璃扶手。這樣設計合法嗎？",
            "🔥 水火不容極限\n\n(測試陽宅三要)": "廚房空間很小，我打算把瓦斯爐直接緊貼著水槽（距離 0 公分），而且瓦斯爐的正對面剛好就是冰箱。聽說這在風水上叫『水火不容』，真的很嚴重嗎？",
            "🚽 中宮造廁\n\n(測試第46條/陽宅十書)": "這是一間老透天，唯一的廁所剛好在房子的『正中央』，而且完全沒有對外窗。我想把它擴建成豪華浴室，如果不移位，只裝一個小抽風機符合法規嗎？風水上會有什麼健康影響？",
            "🏠 穿堂煞與採光\n\n(測試陽宅十書)": "為了讓客廳採光更好，我把大門進來的玄關牆全部打掉，現在一開大門就能直接看到最後面的陽台落地窗，風可以直接灌進來。長輩說這是『穿堂煞』會漏財，真的有這麼誇張？",
            "🛌 樑壓床與通風\n\n(測試魯班經/第46條)": "主臥室天花板有一根深度 80 公分的超大橫樑。設計師建議為了空間感，不要做天花板包覆，直接把床頭放在樑下。這樣睡覺會不會有壓迫感？風水上怎麼說？",
            "🚪 門對門罵門煞\n\n(測試魯班經)": "我家主臥室的門打開，剛好正對著對面小孩房的門，兩扇門距離只有 80 公分。聽說這叫『罵門』會導致家庭失和？如果不能改門的位置，可以用裝修手法化解嗎？"
        }
        def set_prompt(text): st.session_state.prompt_content = text
        for label, text in scenarios.items():
            if st.button(label, use_container_width=True): set_prompt(text)

# ==========================================
# 📝 主介面：輸入與執行
# ==========================================

user_input = st.text_area("需求描述", height=150, key="prompt_content")

design_style = st.selectbox("模擬圖風格", [
    "Modern Minimalist (現代極簡)",
    "Industrial Loft (工業風)",
    "Japanese Wabi-sabi (日式寂侘)",
    "Neo-Chinese (新中式)",
    "Creamy & Cozy (溫潤奶油風)",
    "Scandinavian (北歐簡約風)",
    "Modern Luxury (現代輕奢風)",
    "Vintage Bauhaus (復古包浩斯)",
    "Biophilic Design (自然共生風)",
    "Cyberpunk / Neo-Future (賽博龐克風)"
])
style_en = design_style.split("(")[0].strip()

submit_btn = st.button("🚀 啟動 HarmonyChainSpace")

if submit_btn:
    if not api_key:
        st.error(f"❌ 請先在左側輸入 {provider} API Key 才能啟動大腦！")
        st.stop()
        
    if time.time() - st.session_state.last_request < 5:
        st.warning("⏳ 請勿頻繁操作...")
        st.stop()
    st.session_state.last_request = time.time()

    final_query = user_input + (f"\n(圖片描述：{image_desc})" if image_desc else "")
    
    with st.status("🤖 HarmonyChainSpace 正在協作中...", expanded=True) as status:
        
        st.write("📚 RAG 系統正在翻閱《建築技術規則》與《魯班經》...")
        context_text = core.get_rag_context(final_query)
        
        st.write(f"👷‍♂️ [Agent 1] 建築師正在檢討法規 ({model_name})...")
        try:
            st.session_state.arch_result = core.run_architect_agent(context_text, final_query, provider, api_key, model_name)
        except Exception as e:
            status.update(label="❌ 建築師發生錯誤", state="error")
            st.error(f"建築師錯誤: {e}")
            st.stop()
            
        st.write(f"🔮 [Agent 2] 風水師正在推算吉凶 ({model_name})...")
        try:
            st.session_state.fs_result = core.run_fengshui_agent(context_text, final_query, provider, api_key, model_name)
        except Exception as e:
            status.update(label="❌ 風水師發生錯誤", state="error")
            st.error(f"風水師錯誤: {e}")
            st.stop()

        st.write(f"🤝 [Agent 3] 協調者正在整合方案 ({model_name})...")
        try:
            mediator_json = core.run_mediator_agent(
                st.session_state.arch_result, 
                st.session_state.fs_result, 
                final_query, 
                style_en, 
                provider, 
                api_key, 
                model_name
            )
            st.session_state.mediator_result = mediator_json
        except Exception as e:
            status.update(label="❌ 協調者發生錯誤", state="error")
            st.error(f"協調者錯誤: {e}")
            st.stop()
            
        status.update(label="✅ 協作完成！", state="complete", expanded=False)

if st.session_state.arch_result and st.session_state.fs_result:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="bubble-card sci-bubble"><div class="sci-title">📐 建築科學研究員</div>{st.session_state.arch_result}</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="bubble-card fs-bubble"><div class="fs-title">🔮 風水大師</div>{st.session_state.fs_result}</div>""", unsafe_allow_html=True)

if st.session_state.mediator_result:
    verdict = st.session_state.mediator_result.get("verdict", "")
    img_prompt = st.session_state.mediator_result.get("design_prompt", "")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="bubble-card mediator-bubble"><div class="mediator-title">🤝 協調設計方案</div><p>{verdict.replace(chr(10), '<br>')}</p></div>""", unsafe_allow_html=True)

    if paint_mode != "關閉繪圖" and img_prompt:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎨 AI 模擬效果圖")
        with st.status("🎨 正在繪製模擬圖...", expanded=True) as img_status:
            img = None
            if paint_mode == "Pollinations (免費/無限)":
                st.write("使用 Pollinations (Flux) 引擎...")
                img = core.generate_image_via_pollinations(img_prompt)
            elif paint_mode == "Hugging Face (需Token)":
                st.write("使用 Hugging Face (Flux) 引擎...")
                if not hf_token:
                    st.error("請在左側輸入 Hugging Face Token")
                else:
                    img = core.generate_image_from_hf(img_prompt, hf_token)

            if img:
                img_status.update(label="繪圖完成！", state="complete", expanded=False)
                st.image(img, caption=f"設計模擬圖 ({paint_mode})", use_container_width=True)
                with st.expander("查看 Prompt"): st.code(img_prompt)
            else:
                img_status.update(label="繪圖失敗", state="error")
                st.error("繪圖失敗，請檢查網路或 Token")

    elif paint_mode == "關閉繪圖" and img_prompt:
        with st.expander("查看 AI 生成的繪圖指令 (未執行繪圖)"):
            st.info("已略過繪圖步驟。")
            st.code(img_prompt)
