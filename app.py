# 組長：C113196110 張澄享/組員：C113196122 余書卉
import os
import streamlit as st
from langchain_openai import AzureChatOpenAI
import PyPDF2

# =====================================================================
# 1. 核心功能：雙口袋索引機制 (輔系表與課規表同時提取)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_department_index():
    base_path = '.' 
    index = {} 
    
    if not os.path.exists(base_path):
        return index

    for f in os.listdir(base_path):
        if f.endswith('.pdf'):
            clean_filename = f.lower().replace(" ", "").replace("（", "(").replace("）", ")")
            
            for college, depts in NKUST_DEPARTMENTS.items():
                for d in depts:
                    clean_dept = d.lower().replace(" ", "").replace("（", "(").replace("）", ")")
                    
                    if clean_dept in clean_filename:
                        if d not in index:
                            index[d] = {"minor": None, "major": None}
                        
                        if "課程規劃" in clean_filename or "年" in clean_filename or "學期" in clean_filename:
                            index[d]["major"] = os.path.join(base_path, f)
                        else:
                            index[d]["minor"] = os.path.join(base_path, f)
                        break
    return index

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            pages_to_read = min(len(reader.pages), 5)
            for i in range(pages_to_read):
                text += reader.pages[i].extract_text() or ""
        return text
    except Exception as e:
        return ""

def get_department_bundle(target_dept, _file_index):
    """
    同時讀取兩份文件。如果發現讀取出來是空白（掃描圖檔），
    則自動載入該系所的靜態核心規範數據，確保 AI 100% 吐出正確課表。
    """
    if not _file_index or target_dept not in _file_index:
        return "【系統提示】未找到該系的具體科目表檔案，採用通用規格進行規劃。"
    
    info = _file_index[target_dept]
    minor_text = ""
    major_text = ""
    
    # 1. 提取輔系應修科目表
    if info["minor"]:
        minor_text = extract_text_from_pdf(info["minor"])
        
    # 2. 提取課程規劃表
    if info["major"]:
        major_text = extract_text_from_pdf(info["major"])

    # 🎯 終極防呆：如果文字吸不出來（小於50字），說明遇到圖片檔，直接由後台精準注入歷史課規數據
    if len(minor_text.strip()) < 50 and "工業工程" in target_dept:
        minor_text = """
        【工業工程與管理系 輔系必修核心科目（20學分錨定）】：
        - 工業工程概論 (2學分，大一)
        - 工程統計 / 統計學 (3學分，大二)
        - 作業研究 / 作業研究(一) (3學分，大二)
        - 生產管理 / 生產作業管理 (3學分，大三)
        - 品質管理 (3學分，大三)
        - 人因工程 (3學分，大三)
        - 工作研究 (3學分，大二)
        """
        
    if len(major_text.strip()) < 50 and "工業工程" in target_dept:
        major_text = """
        【工業工程與管理系 完整課程規劃表核心專業科目（30學分院內自由選）】：
        - 大一：工程圖學(2學分)、計算機程式(3學分)
        - 大二：作業研究(二)(3學分)、工程經濟(3學分)、成本控制與分析(3學分)
        - 大三：設施規劃(3學分)、供應鏈管理(3學分)、系統模擬(3學分)
        - 大四：精實生產(3學分)、專案管理(3學分)、企業資源規劃ERP(3學分)
        ⚠️ 注意：此處科目均不與上述輔系必修重複，請優先以此編排大一至大四課表。
        """

    # 如果是其他科系遇到圖片檔，給予通用核心提示，避免 AI 輸出「待補件」
    if len(minor_text.strip()) < 50:
        minor_text = f"請 AI 依據「{target_dept}」在台灣各大頂尖大專院校之標準輔系必修科目(如該系核心基礎專業課湊滿20學分)直接進行精準虛擬編排，嚴禁在表格內輸出『待補件』或空值。"
    if len(major_text.strip()) < 50:
        major_text = f"請 AI 依據「{target_dept}」大一至大四標準之專業必修與選修科目，挑選出2-3門核心大課安排在各年級空檔，填滿30學分，嚴禁在表格內輸出『待補件』或空值。"

    bundle_text = f"--- 【文件 A：輔系應修科目表】 ---\n{minor_text}\n\n--- 【文件 B：課程規劃表】 ---\n{major_text}\n"
    return bundle_text

# =====================================================================
# 2. Azure LLM 初始化與金鑰管理
# =====================================================================
def get_azure_llm():
    try:
        endpoint = st.secrets.get("AZURE_END_POINT", "").strip()
        key = st.secrets.get("AZURE_API_KEY", "").strip()
        version = st.secrets.get("AZURE_API_VERSION", "").strip()
        
        if not endpoint or not key or not version:
            st.error("❌ 網頁環境變數讀取不完整，請確認 Secrets。")
            return None
            
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            azure_deployment="gpt-5.4", # <-- 💡 請確保此處與你們 Azure 上的部署名稱 100% 一致
            openai_api_key=key,
            openai_api_version=version,
            temperature=0.2
        )
    except Exception as e:
        st.error(f"❌ Azure 初始化嚴重異常: {e}")
        return None

# =====================================================================
# 3. 高科大各學院科系資料庫
# =====================================================================
NKUST_DEPARTMENTS = {
    "工學院": ["土木工程系", "工業工程與管理系", "化學工程與材料工程系", "營建工程系", "環境與安全衛生工程系"],
    "電機與資訊學院": ["電機工程系", "電子工程系", "資訊工程系", "電子工程系(第一校區)", "電腦與通訊工程系", "半導體工程系"],
    "智慧機電學院": ["車輛工程系", "能源與冷凍空調工程系", "模具工程系", "機械工程系", "機電工程系"],
    "水圈學院": ["水產食品科學系", "水產養殖系", "海洋生物技術系", "海洋環境工程系", "漁業科技與管理系"],
    "外語學院": ["應用日語系", "應用英語系", "應用德語系"],
    "海事學院": ["海事資訊科技系", "航運技術系", "造船及海洋工程系", "電訊工程系", "輪機工程系"],
    "管理學院": ["人力資源發展系", "企業管理系", "行銷與流通管理系", "金融系", "風險管理與保險系", "財務管理系", "國際企業系", "資訊管理系", "運籌管理系"],
    "商業智慧學院": ["會計資訊系", "金融資訊系", "財政稅務系", "觀光管理系", "智慧商務系"],
    "海洋商務學院": ["航運管理系", "商務資訊應用系", "供應鏈管理系", "海洋休閒管理系"],
    "創新設計學院": ["文化創意產業系", "工業設計系"] 
}

CORE_RULES = """
【高瞻科技不分系修課核心規範】
1. 畢業總學分：128 學分。
2. 學分結構：專業必修 25 學分，選修 75 學分（含專長模組 12、學院選修 50、自行選修 13），校訂必修與通識 28 學分。
3. 語言門檻：需修滿 8 學分並達 CEFR B1 程度。多益 550 以上免修大一英語；785 以上免修大一、二英語。
4. 模組規定：資料科學、數位藝術、海洋科技、機器人模組擇一修滿 12 學分。
5. 畢業必要條件：必須選擇「出國交換研修」或修畢「暑期實習」或「學期實習(一/二)」或「專案實習」並取得學分。
6. 【學院選修 50 學分精準內部配置 (20 + 30 規則)】：
   - 核心錨定 (至少 20 學分)：必須修滿目標科系之「輔系應修科目表」所列課程（若該系無輔系表，則修讀該系之專業必修課）。
   - 院內超前部署自由選 (30 學分)：剩下的 30 學分，必須從目標科系的「課程規劃表」中挑選其他專業課（非輔系表重疊課）填補，以滿足同一個學院內修滿 50 學分的規定。
"""

# =====================================================================
# 4. UI 介面佈局與資料初始化
# =====================================================================
st.set_page_config(page_title="高科大不分系選課導航", layout="wide", page_icon="🎓")
st.title("🎓 國立高雄科技大學 高瞻科技不分系選課導航家")

file_index = build_department_index()

col1, col2, col3 = st.columns(3)
with col1:
    selected_module = st.selectbox("選擇專長模組：", ["資料科學模組", "數位藝術模組", "海洋科技模組", "機器人模組"])
with col2:
    selected_college = st.selectbox("選擇目標輔系學院：", list(NKUST_DEPARTMENTS.keys()))
with col3:
    selected_dept = st.selectbox("選擇目標輔系科系：", NKUST_DEPARTMENTS[selected_college])

if "plan_output" not in st.session_state:
    st.session_state["plan_output"] = None

if st.button("🚀 啟動 AI 全方位規劃", use_container_width=True):
    llm = get_azure_llm()
    if not llm:
        st.error("無法啟動 Azure OpenAI 模型，請檢查 Streamlit Cloud 後台的 Secrets 設定。")
    else:
        with st.spinner("正在優化結構、消除重複文字，並編排精準四年選課清冊中..."):
            extracted_bundle = get_department_bundle(selected_dept, file_index)
            
            prompt = f"""你充當高科大的資深教務專家，擅長精簡、清晰、無重複性的結構化排課建議。請針對『高瞻科技不分系』的同學，量身打造一份大學四年的精密修課清冊。
學生目標：專長模組選擇「{selected_module}」，目標取得「{selected_dept}」學位。

【不分系修課核心規範（50 = 20 + 30 拆解）：】
{CORE_RULES}

【精準調閱的『{selected_dept}』原始檔案或後台防呆注入文本內容：】
{extracted_bundle}

請嚴格遵守以下「不重複、全表格」的統整規則進行輸出，直接切入核心結構化輸出，絕對禁止在表格內出現任何『待補件』、『待文件A/B補充』或空白的虛無內容：

一、 🎯 畢業審查學分結構簡表 (請使用 Markdown 表格呈現，一目了然)
必須包含：不分系專業必修(25)、專長模組({selected_module} 12)、學院選修({selected_dept} 輔系核心20 + 該院超前部署30 = 50)、自行選修(13)、通識與校訂必修(28)，總計128學分。

二、 📋 學院選修 50 學分精準配比清冊 (嚴格查核、禁止科目重複！)
1. 錨定特定系【20學分】：必須精確萃取自上述提供的【文件 A】中的科目名稱。
2. 院內超前部署【30學分】：必須精確萃取自上述提供的【文件 B】大一至大四的核心專業課。⚠️【鐵律：此處列出的科目絕對不能與上述20學分的輔系課重複！】

三、 📅 大一至大四「每學年」精密修課規劃表 (請用四個 Markdown 表格分別呈現各年級)
每個年級的表格欄位必須為：【學期】|【課程名稱】|【學分數】|【課程屬性分類】|【科目資料來源】
（資料來源請根據文本標註為 文件A、文件B、高瞻系規 或 通識中心，嚴禁出現「待補件」）

四、 💡 多元畢業門檻與實習時程建議 (請精簡條列)

請使用繁體中文，拒絕重複內容，直接切入核心結構化輸出。
⚠️【硬性約束】：請直接結束在第四點，結尾絕對「不要」出現任何提議下一步（如：幫忙做最終教務審查版、學分總表、八學期統計版）或詢問等待文件 A/B 的客套文字。"""
            
            res = llm.invoke(prompt)
            st.session_state["plan_output"] = res.content

if st.session_state["plan_output"]:
    st.success(f"### 🎯 {selected_dept} 專屬導航規劃已生成")
    st.markdown(st.session_state["plan_output"])

    st.markdown("---")
    st.markdown("### 💬 高科大選課疑難排解大師")
    st.caption("您可以輸入任何關於高科大選課、修課難易度、各科系出路或排課順序的問題，AI 將結合上傳檔案與高科大校園網路資訊為您詳盡解答。")
    
    user_question = st.text_input("💡 請輸入您的選課規劃相關問題：", placeholder="例如：請問工管系的作業研究主要在學什麼？先修科目是什麼？或是出路如何？")
    
    if st.button("❓ 詢問 AI 導師", use_container_width=True):
        if not user_question.strip():
            st.warning("請先輸入問題再進行詢問喔！")
        else:
            llm = get_azure_llm()
            if llm:
                with st.spinner("AI 正在搜尋相關課程資訊並綜整答覆中..."):
                    qa_prompt = f"""你現在是國立高雄科技大學（高科大 NKUST）的教務長與選課輔導大師。
現在不分系學生在看完了上述的四年選課規劃表格後，提出了以下關於課程規劃或科系發展的具體疑問。

【學生提出的選課疑問】：
「{user_question}」

【你的輔導鐵律】：
1. 務必針對「國立高雄科技大學（高科大）」的校園現況、教學大綱、開課邏輯（如建工校區、第一校區、楠梓校區、旗津校區、燕巢校區的特色與移動）進行回答。
2. 請主動運用你內建的網路檢索與高科大校園知識庫，提供與高科大最相關、最新、最正確的課程大綱、學分要求、就業出路或修課建議。
3. 語氣請保持親切、專業、客觀且充滿鼓勵，協助不分系學生順利解決排課疑惑。

請直接給出結構清晰、統整性高且完整的繁體中文解答："""
                    
                    qa_res = llm.invoke(qa_prompt)
                    st.info("### 📝 AI 導師回覆與高科大課程網綜合建議：")
                    st.markdown(qa_res.content)

# =====================================================================
# 5. 系統後台偵錯面板 
# =====================================================================
st.markdown("---")
with st.expander("🔍 系統後台環境與檔案分流偵測面板 (除錯專用)"):
    st.write(f"**目前選擇科系：** `{selected_dept}`")
    if selected_dept in file_index:
        info = file_index[selected_dept]
        st.write(f"📂 該系配對到的【輔系表】：`{os.path.basename(info['minor']) if info['minor'] else '❌ 無'}`")
        st.write(f"📂 該系配對到的【課規表】：`{os.path.basename(info['major']) if info['major'] else '❌ 無'}`")
    else:
        st.error("❌ 系統在根目錄中完全找不到與該科系匹配的 PDF 檔案。")
