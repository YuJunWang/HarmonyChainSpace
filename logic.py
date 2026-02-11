import os
import base64
import urllib.parse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

# 引入各大模型庫
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# ===========================
# 1. 知識庫
# ===========================
class KnowledgeBase:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.vector_db = None

    def load_db(self):
        if os.path.exists(self.persist_dir) and os.path.isdir(self.persist_dir):
            try:
                self.vector_db = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
                return True
            except Exception as e:
                print(f"❌ 資料庫載入失敗: {e}")
                return False
        else:
            print(f"⚠️ 警告：找不到路徑 {self.persist_dir}，請先執行資料重建腳本！")
            return False

    def get_context(self):
        if not self.vector_db:
            success = self.load_db()
            if not success: return None
        return self.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20}
        )

# ===========================
# 2. 定義輸出結構
# ===========================
class MediatorOutput(BaseModel):
    verdict: str = Field(description="協調者的最終折衷方案與建議，純文字")
    design_prompt: str = Field(description="給 AI 繪圖模型的英文 Prompt")

# ===========================
# 3. 多代理人核心邏輯
# ===========================
class LogicCore:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.kb.load_db()

    # --- 🏭 LLM 工廠 ---
    def _create_llm(self, provider, api_key, model_name=None):
        if not api_key:
            raise ValueError(f"請輸入 {provider} 的 API Key")

        if provider == "Groq":
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name or "llama-3.3-70b-versatile",
                temperature=0.5
            )
        elif provider == "OpenAI":
            return ChatOpenAI(
                api_key=api_key,
                model=model_name or "gpt-4o",
                temperature=0.5
            )
        elif provider == "Gemini":
            return ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model_name or "gemini-2.5-flash",
                temperature=0.5
            )
        else:
            raise ValueError("不支援的模型供應商")

    # --- 📚 共用功能：RAG 檢索 ---
    def get_rag_context(self, query):
        retriever = self.kb.get_context()
        docs = retriever.invoke(query) if retriever else []
        context_text = "\n".join([d.page_content for d in docs])
        return context_text

    # --- 👁️ 視覺代理人 ---
    def analyze_image(self, image_bytes, api_key, mime_type="image/jpeg"):
        if not api_key: return "未提供 Gemini Key，無法分析圖片。"

        vision_model = ChatGoogleGenerativeAI(
            model="gemini-3.0-pro",
            temperature=0.7,
            google_api_key=api_key
        )
        b64_string = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{b64_string}"

        prompt = "請仔細觀察這張室內設計圖片。描述空間佈局，重點包含：窗戶位置、樑柱位置、門的相對位置。請用客觀的純文字描述。"
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": image_url}
                ]
            }
        ]

        response = vision_model.invoke(message)
        return response.content

    # --- 🏗️ Agent 1: 建築科學家 ---
    def run_architect_agent(self, context, query, provider, api_key, model_name):
        llm = self._create_llm(provider, api_key, model_name)

        template = """
        你是一位嚴謹、理性的資深註冊建築師。你的職責是確保空間設計符合《建築技術規則》、人體工學與居住安全。

        【任務目標】
        根據下方提供的【參考資料】，以科學務實的方案評估使用者的設計需求在法規與物理層面是否可行。

        【嚴格約束】
          1. **你是唯物主義者**：你完全不相信風水、氣場、運勢或鬼神之說。
          2. **過濾雜訊**：【參考資料】中可能混雜了風水古籍（如魯班經、煞氣等），請將這些視為「無效雜訊」並**完全忽略**，不要在回答中提及，也不要反駁，直接無視。
          3. **依據法規**：回答必須引用具體的法規條文（如：建築技術規則第XX條）或具體的人體工學數據（如：走道寬度至少90cm）。
          4. **語氣**：專業、客觀、冷靜，只談數據與法律。

        【參考資料】
        {context}

        【使用者需求】
        {question}

        請以建築師的角度進行分析，執行以下思考步驟：
        1. **設計意圖分析**：理解使用者想做什麼。
        2. **科學性的判斷**：
           - 思考改動對空間造成的物理結果（油煙、濕氣、通風）。
           - 檢查【參考資料】，僅引用「直接相關」條文。
           - ⚠️ 重要：若參考資料無關，請忽略並以科學常識回答。

        3. **輸出要求**：請輸出 HTML 格式，包含：
           - `<p><b>📐 設計意圖：</b>...</p>`
           - `<p><b>🔍 科學檢視與法規：</b>...</p>`
           - `<ul><li><b>引用依據：</b>(直接寫出相關法條，若無RAG資料則寫"依據一般建築常規")</li><li><b>具體建議：</b>...</li></ul>`
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    # --- 🔮 Agent 2: 風水大師 ---
    def run_fengshui_agent(self, context, query, provider, api_key, model_name):
        llm = self._create_llm(provider, api_key, model_name)

        template = """
        你是一位傳承千年的風水堪輿大師。你的職責是依據《魯班經》、《陽宅三要》等古籍，診斷空間中的「形煞」與「氣場」。

        【任務目標】
        根據下方提供的【參考資料】，以傳統風水、玄學的角度評估使用者的設計需求是否存在風水禁忌或對居住者心理的負面影響。

        【嚴格約束】
        1. **你是玄學家**：你關注的是「氣」、「煞」、「五行」與「心理暗示」。
        2. **忽略法規**：不要管建築法規是否允許，也不要管結構是否安全（那是建築師的事）。即便資料中有法規條文，也請忽略。
        3. **專注禁忌**：如果設計觸犯了禁忌（如穿堂煞、樑壓床、水火不容），請直言不諱地指出其後果（如漏財、血光、口角）。
        4. **語氣**：傳統、帶有警示性、關注居住者的運勢與健康。

        【參考資料】
        {context}

        【使用者需求】
        {question}

        請以風水師的角度進行推算，執行以下思考步驟：
        1. **煞氣診斷**：判斷格局涉及哪種具體禁忌。
        2. **古籍比對**：
           - 檢查【參考資料】是否有對應原文。
           - ⚠️ 重要：若參考資料無關，請忽略並以內建風水知識回答。

        3. **輸出要求**：請輸出 HTML 格式，包含：
           - `<p><b>🔮 陽宅格局診斷：</b>...</p>`
           - `<p><b>📜 古籍與民俗觀點：</b>...</p>`
           - `<ul><li><b>經典引據：</b>(直接寫出經典名稱，若無RAG資料則寫依據一般風水見解)...</li><li><b>化解之道：</b>...</li></ul>`
        """

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    # --- 🤝 Agent 3: 協調設計師 ---
    def run_mediator_agent(self, arch_res, fs_res, query, style, provider, api_key, model_name):
        llm = self._create_llm(provider, api_key, model_name)
        parser = JsonOutputParser(pydantic_object=MediatorOutput)

        template = """
        你是一位擅長跨領域整合的資深室內設計總監。你剛聽取了兩位專家的意見：
        1. **建築師**：關注法規與安全（理性）。
        2. **風水師**：關注氣場與禁忌（感性）。

        【使用者需求】：{query}
        【風格偏好】：{style}

        【建築師意見】：{arch_res}
        【風水師意見】：{fs_res}

        請完成以下兩項任務並輸出 JSON：

        ### 任務 1：協調與折衷 (Verdict)
        1. 綜合上述兩位的觀點，指出衝突點，或是組合雙方提出的共識。
            a. **物理上可行**：符合建築師提出的法規與安全底線。
            b. **心理上舒適**：透過設計手法（如屏風、色彩、造型、燈光）化解風水師提出的疑慮（化煞）。
        2. 提供一個「具體折衷方案」，同時滿足科學（通風/採光）與風水（心理/避煞）。
        3. 輸出純文字，語氣溫和專業。

        ### 任務 2：視覺化 Prompt 生成 (Design Prompt)
        根據你的折衷方案與使用者的「{style}」風格，撰寫一個英文 Prompt。
        包含詳細的結構關係、環境的描寫、空間中物品的相對關係、色彩與材質等細節
        **關鍵要求：**
        - **針對 FLUX 模型優化**
        - **視角設定**：使用 "High-angle 3/4 perspective view" 及 "Cutaway 3D render"。
        - **構圖目標**：畫面稍微旋轉，不要完全正對牆面。
        - 包含風格關鍵字 (e.g., photorealistic, 8k)。
        - 避免不符合現實的景象。

        {format_instructions}
        """

        prompt = ChatPromptTemplate.from_template(template, partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = prompt | llm | parser
        return chain.invoke({"arch_res": arch_res, "fs_res": fs_res, "query": query, "style": style})

    # --- 🎨 繪圖工具 ---
    def generate_image_via_pollinations(self, prompt):
        """
        使用 Pollinations.ai 免費生成圖片
        """
        import urllib.parse
        import random  # 新增 random 以生成隨機種子

        try:
            # 1. 對 Prompt 進行 URL 編碼，避免特殊字符導致連結失效
            encoded_prompt = urllib.parse.quote(prompt)

            # 2. 生成隨機種子 (Seed)，確保每次生成的構圖都不一樣
            seed = random.randint(0, 99999)

            # 3. 建立 URL
            # model=flux-realism: 指定 2025 更強的真實感模型
            # seed={seed}: 固定隨機性
            # nologo=true: 嘗試隱藏浮水印
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?model=flux-realism&width=1024&height=768&seed={seed}&nologo=true"

            return url
        except Exception as e:
            print(f"Pollinations Error: {e}")
            return None

    def generate_image_from_hf(self, prompt, hf_token):
        if not hf_token: return None
        from huggingface_hub import InferenceClient
        client = InferenceClient(model="black-forest-labs/FLUX.1-schnell", token=hf_token)
        try:
            image = client.text_to_image(prompt)
            return image
        except Exception as e:
            print(f"HF Error: {e}")
            return None
