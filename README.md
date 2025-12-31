# ☯️ HarmonyChainSpace · 宅運科學雙軌協作系統

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Multi--Agent-green)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Bridging the gap between Architectural Science and Feng Shui Metaphysics."**
> 當理性的建築法規，遇上感性的風水哲學——透過 AI 多代理人系統，尋求空間設計的最佳平衡點。

[![Watch Presentation](https://img.shields.io/badge/View-Presentation-orange?style=for-the-badge&logo=html5)](https://yujunwang.github.io/HarmonyChainSpace/)

## 📖 專案簡介 (Introduction)

**HarmonyChainSpace** 是一個基於 **LangChain** 框架開發的 **多代理人協作系統 (Multi-Agent System)**。

在室內設計實務中，「科學法規」與「傳統風水」常發生衝突（例如：為了通風想拆牆，卻犯了穿堂煞）。本專案模擬了專家會診的流程，利用不同角色的 AI Agent 分別引用權威資料庫（RAG），進行邏輯辯證，最終由協調者提出兼顧安全與心理舒適的折衷方案，並自動生成視覺化設計圖。

## 🧠 核心架構 (System Architecture)

本系統採用 **三方辯證 (Dialectic Triad)** 架構：

### 1. 👷‍♂️ Agent A: 建築科學家 (The Architect)
* **核心邏輯**：基於物理環境、人體工學、法律規範。
* **知識庫 (RAG)**：引用《建築技術規則》與現代人體工學數據。
* **任務**：檢視空間的安全性、採光通風、動線合理性與適法性。

### 2. 🔮 Agent B: 風水諮詢師 (The Feng Shui Master)
* **核心邏輯**：基於氣場流動、心理暗示、傳統民俗禁忌。
* **知識庫 (RAG)**：引用《魯班經》、《陽宅三要》等古籍觀點。
* **任務**：診斷形煞（如穿堂煞、壁刀煞）、五行方位與居住者心理影響。

### 3. 🤝 Agent C: 協調設計師 (The Mediator)
* **核心邏輯**：衝突解決 (Conflict Resolution) 與設計整合。
* **任務**：
    1.  分析 Agent A 與 Agent B 的觀點差異。
    2.  提出「以科學手段化解風水顧慮」的折衷方案。
    3.  **Prompt Engineering**：將方案轉化為針對 **Flux** 模型優化的繪圖指令。

## 📊 資料來源與聲明 (Data Sources & Disclaimer)

本系統的 RAG 知識庫由兩個不同來源的資料集構成，特此說明資料取得方式與適用範圍：

### 1. 建築法規資料庫 (Real-world Data)
* **來源**：[全國法規資料庫 (Laws & Regulations Database of R.O.C.)](https://law.moj.gov.tw/)
* **取得方式**：使用 Python 爬蟲直接抓取最新的《建築技術規則》與相關修法條文。
* **準確度**：高。系統會優先引用具體的法條編號（如：建築技術規則建築設計施工編第 33 條）。

### 2. 風水玄學資料庫 (Synthetic Data)
* **來源**：網際網路公開之風水案例與民俗討論。
* **取得方式**：由於古籍（如魯班經、陽宅三要）的數位化文本格式不一且難以直接向量化，本專案採用 **LLM Augmented Generation (合成資料)** 技術。
    * 我們收集了常見的現代風水案例。
    * 使用 LLM 模擬古籍口吻進行改寫與結構化，生成適合 RAG 檢索的「合成古籍資料」。
* **準確度**：僅供民俗參考。此部分旨在模擬風水師的思維邏輯，不代表絕對的吉凶定論。

## 📂 資料工程 (Data Engineering)

本專案包含完整的資料處理管線 (Data Pipeline)，相關程式碼位於 `docs/` 資料夾中：

* **`docs/HarmonyChainSpace_processing.ipynb`**: 
    * 包含從法規網站爬取資料的 Crawler 實作。
    * 包含使用 LLM 生成風水合成資料的 Prompt Engineering 過程。
    * 展示如何使用 `Sentence-Transformers` 進行 Embedding 並寫入 ChromaDB 的完整流程。

## 🛠️ 技術棧 (Tech Stack)

* **Framework**: [Streamlit](https://streamlit.io/) (Interactive UI)
* **Orchestration**: [LangChain](https://python.langchain.com/) (Chain of Thought, Tool Use)
* **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Static Embedding Deployment)
* **LLM Inference** (Dynamic Switching):
    * ⚡ **Groq** (Llama-3.3-70B): Ultra-fast reasoning.
    * 🧠 **OpenAI** (GPT-4o): High-intelligence logic.
    * 🐢 **Google** (Gemini 2.5 Flash): Long-context processing.
* **Visual Understanding**: **Gemini 3.0 Pro** (Multi-modal Image Analysis).
* **Image Generation**: **Pollinations.ai** (Flux Realism) / Hugging Face Inference API.

## 🚀 快速開始 (Quick Start)

### 1. Clone 專案
```bash
git clone https://github.com/YuJunWang/HarmonyChainSpace.git
cd HarmonyChainSpace
```
### 2. 建立環境 (建議 Python 3.11)
```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate # macOS / Linux: source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```
### 3. 啟動應用
```bash
streamlit run app.py
```
### 4. 設定 API Key
啟動後，請在網頁左側 Sidebar 輸入您的 API Key (Groq, OpenAI, 或 Gemini) 即可開始體驗。
* 目前Pollinations (Flux)似乎無法再使用免費繪圖了。

## 📂 專案結構
```text
HarmonyChainSpace/
├── app.py              # 前端介面與 Agent 調度邏輯
├── logic.py            # LLM Factory, RAG 檢索與 Tool 定義
├── styles.py           # CSS 樣式優化
├── requirements.txt    # 相依套件清單
├── .python-version     # 指定部署環境 Python 版本 (3.11)
├── chroma_db/          # 預訓練向量資料庫 (Pre-indexed)
├── docs/               # 資料工程與實驗筆記\資料爬取與 Embedding 腳本
│   └── HarmonyChainSpace_processing.ipynb
└── README.md           # 說明文件
```
## 📄 License

本專案採用 **MIT License** 授權 - 詳細請參閱 [LICENSE](LICENSE) 檔案。

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 作者 (Author)
**王譽鈞 (YuJunWang)**

* Data Engineer / Data Scientist / AI-Augmented Developer 

* [GitHub Profile](https://github.com/YuJunWang)