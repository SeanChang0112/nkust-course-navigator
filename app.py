import os
import streamlit as st
from langchain_openai import AzureChatOpenAI
import PyPDF2

# =====================================================================
# 1. 核心功能：直接讀取 GitHub 最外層平鋪的 PDF 規範檔案 (雙口袋索引)
# =====================================================================

@st.cache_data(show_spinner=False)
def build_department_index():
    """
    掃描 GitHub 根目錄，將檔案精準分類為「輔系表」與「課規表」，
    徹底解決多檔案混淆導致 AI 讀錯的問題。
    """
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
        return f"(檔案讀取失敗: {e})"

@st.cache_data(show_spinner=False)
def get_department_requirements_optimized(target_dept, _file_index):
    if not _file_index or target_dept not in _file_index:
        if "車輛" in target_dept or "能源" in target_dept or "冷凍" in target_dept:
            return f"【教務系統提示】該系未訂定獨立輔系標準。依據高瞻系規，AI 將依據該系的『專業必修科目結構表』進行審查，該生需修讀該系專業必修科目達 20 學分以符合學位授予資格。"
        return "【系統提示】未找到該系的具體科目表檔案，系統目前將採用校級通則進行合規性規劃。"
    
    dept_files = _file_index[target_dept]
    
    if dept_files["minor"]:
        file_path = dept_files["minor"]
        return f"【系統已精準調閱輔系應修科目表：{os.path.basename(file_path)}】\n\n" + extract_text_from_pdf(file_path)
    
    elif dept_files["major"]:
        file_path = dept_files["major"]
        return f"【系統已精準調閱課程規劃表：{os.path.basename(file_path)}】\n\n" + extract_text_from_pdf(file_path)
        
    return "【系統提示】未找到該系的具體科目表檔案，系統目前將採用校級通則進行合規性規劃。"

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
            azure_deployment="gpt5.4", 
            openai_api_key=key,
            openai_api_version=version,
            temperature=0.3
        )
    except Exception as e:
        st.error(f"❌ Azure 初始化嚴重異常: {e}")
        return None

# =====================================================================
# 3. 高科大各學院科系資料庫與「50學分極致清晰版」規範 (優化重點)
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

# 🎯 這裡進行了精準的條款重寫，明確約束 50 學分的內部配置 (20 + 30 規則)
CORE_RULES = """
【高瞻科技不分系修課核心規範】
1. 畢業總學分：128 學分。
2. 學分結構：專業必修 25 學分，選修 75 學分（含專長模組 12、學院選修 50、自行選修 13），校訂必修與通識 28 學分。
3. 語言門檻：需修滿 8 學分並達 CEFR B1 程度。多益 550 以上免修大一英語；785 以上免修大一、二英語。
4. 模組規定：資料科學、數位藝術、海洋科技、機器人模組擇一修滿 12 學分。
5. 畢業必要條件：必須選擇「出國交換研修」或修畢「暑期實習」或「學期實習(一/二)」或「專案實習」並取得學分。
6. 【超級硬性規定】學院選修 50 學分的精準結構拆解如下：
   - 學生必須在目標學院內修滿【50學分】（共同教育學院除外）。
   - 核心錨定(至少20學分)：學生必須指定該學院中的「任一目標科系」，並修滿該科系之「輔系課程至少 20 學分」。
   - 特殊例外：若該目標科系未訂定輔系標準（例如：車輛工程系、能源與冷凍空調系），則改為修滿該科系的「專業必修課程至少 20 學分」。
   - 剩餘院內游擊(餘下30學分)：扣除上述特定系的 20 學分後，剩下的【30學分】，學生可以自由選修「該學院內任一系所」開設的任何專業課程來湊滿。
   - 滿足以上條件，畢業時方可授予該學院該學系所屬之學士學位。
"""

# =====================================================================
# 4. UI 介面佈局與資料初始化
# =====================================================================
st.set_page_config(page_title="高科大不分系選課導航", layout="wide", page_icon="🎓")
st.title("🎓 高瞻科技不分系選課導航家 (雲端永久運行版)")

file_index = build_department_index()

col1, col2, col3 = st.columns(3)
with col1:
    selected_module = st.selectbox("選擇專長模組：", ["資料科學模組", "數位藝術模組", "海洋科技模組", "機器人模組"])
with col2:
    selected_college = st.selectbox("選擇目標輔系學院：", list(NKUST_DEPARTMENTS.keys()))
with col3:
    selected_dept = st.selectbox("選擇目標輔系科系：", NKUST_DEPARTMENTS[selected_college])

if st.button("🚀 啟動 AI 全方位規劃", use_container_width=True):
    llm = get_azure_llm()
    if not llm:
        st.error("無法啟動 Azure OpenAI 模型，請檢查 Streamlit Cloud 後台的 Secrets 設定。")
    else:
        with st.spinner("正在依據 50+20 核心法規與隨附檔案進行極速選課規劃中..."):
            extracted_content = get_department_requirements_optimized(selected_dept, file_index)
            
            prompt = f"""你是一位高科大的資深教務導師。請針對『高瞻科技不分系學士學位學程』的同學進行大學四年的完整修課路徑規劃與畢業學分審查建議。
使用者目前設定之目標：專長模組選擇「{selected_module}」，並期望依據法規取得「{selected_dept}」之學士學位。

【高瞻不分系核心修課規範與畢業審查要點（⚠️請嚴格遵守第6點之 50 = 20 + 30 的學分拆解規則）：】
{CORE_RULES}

【系統精準檢索到的目標系所規範文件內容（請依據此文件內列出的具體科目進行排課）：】
{extracted_content}

請根據上述提供的具體規範與文件，進行全方位的交叉比對與修課安排：
1. 畢業學分組成結構：精確說明該生如何滿足 128 總學分（包含專業必修 25、模組選修 12、學院選修 50、自行選修 13、校訂必修與通識 28）。
2. 【核心重點】學院選修 50 學分的具體實踐配比：明確向學生列出，在學院選修的 50 學分中，哪【20學分】是用來修讀「{selected_dept}」的輔系課（或專業必修課），並具體列出該系隨附文件中的科目名稱；另外剩下的【30學分】該院內自由選修課程，建議可以修讀哪些相關系所的課程。
3. 四年修課路徑發展建議：具體列出大一至大四各學年的修課策略，必須將該生選擇的「{selected_module}」核心課程、特定科系的 20 學分核心課、以及院內其餘 30 學分的游擊課完美融合進各年級的排課中。
4. 多元選修與語言門檻提醒：針對實習/出國交換的硬性畢業條件，以及多益免修學分或補課的機制，給予具體的執行時程建議。

請使用繁體中文，以專業、詳盡且對學生充滿鼓勵的親切語氣回答。"""
            
            res = llm.invoke(prompt)
            st.success(f"### 🎯 {selected_dept} 專屬導航規劃已生成")
            st.markdown(res.content)

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
