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
    """
    掃描 GitHub 根目錄，將檔案精準分類為「輔系表」與「課規表」，
    讓系統可以同時撈出兩份檔案餵給 AI。
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
            # 限制讀取前 5 頁核心課表內容
            pages_to_read = min(len(reader.pages), 5)
            for i in range(pages_to_read):
                text += reader.pages[i].extract_text() or ""
        return text
    except Exception as e:
        return f"(檔案讀取失敗: {e})"

def get_department_bundle(target_dept, _file_index):
    """
    同時讀取輔系應修科目表與課程規劃表，打包成超大文本餵給 AI。
    """
    if not _file_index or target_dept not in _file_index:
        if "車輛" in target_dept or "能源" in target_dept or "冷凍" in target_dept:
            return "【教務系統提示】該系未訂定獨立輔系標準。依據高瞻系規，AI 將依據該系的『專業必修科目結構表』進行審查，該生需修讀該系專業必修科目達 20 學分以符合學位授予資格。"
        return "【系統提示】未找到該系的具體科目表檔案，系統目前將採用校級通則進行合規性規劃。"
    
    info = _file_index[target_dept]
    bundle_text = ""
    
    # 1. 讀取輔系表
    if info["minor"]:
        bundle_text += f"--- 【文件 A：輔系應修科目表 (真實檔名: {os.path.basename(info['minor'])})】 ---\n"
        bundle_text += extract_text_from_pdf(info["minor"]) + "\n\n"
    else:
        bundle_text += "--- 【文件 A：輔系應修科目表】 --- (該系未訂定獨立輔系標準，改由專業必修審查)\n\n"
        
    # 2. 讀取課程規劃表
    if info["major"]:
        bundle_text += f"--- 【文件 B：全系各年級完整課程規劃表 (真實檔名: {os.path.basename(info['major'])})】 ---\n"
        bundle_text += extract_text_from_pdf(info["major"]) + "\n\n"
        
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
            azure_deployment="gpt-5.4", # <-- 💡 請確保此處與你們 Azure 上的部署名稱 100% 一致 (例如 gpt35 或 gpt-35-turbo)
            openai_api_key=key,
            openai_api_version=version,
            temperature=0.3
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
   - 錨定特定系 (20 學分)：必須修滿目標科系之「輔系課程至少 20 學分」（若該系無輔系表，則修該系專業必修 20 學分）。
   - 院內超前部署自由選 (30 學分)：剩下的 30 學分，學生應利用自身的修課空檔，盡可能提前選修該目標科系「課程規劃表」上所開設的其他專業必修或核心選修課程（即那些沒有被列在輔系表上、但出現在各年級課程規劃表中的系專業課），以此來填滿這 30 學分，達成深度專業化。
"""

# =====================================================================
# 4. UI 介面佈局與資料初始化
# =====================================================================
st.set_page_config(page_title="高科大不分系選課導航", layout="wide", page_icon="🎓")
st.title("🎓 高瞻科技不分系選課導航家")

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
        with st.spinner("正在執行雙文件深度交叉檢索並編排高精準度四年選課路徑中..."):
            # 同時抓取輔系表和課程規劃表的文字內容
            extracted_bundle = get_department_bundle(selected_dept, file_index)
            
            prompt = f"""你是一位高科大的資深教務導師。請針對『高瞻科技不分系學士學位學程』的同學進行大學四年的完整修課路徑規劃與畢業學分審查建議。
使用者目前設定之目標：專長模組選擇「{selected_module}」，並期望依據法規取得「{selected_dept}」之學士學位。

【高瞻不分系核心修課規範與畢業審查要點（⚠️請特別注意 50 = 20 + 30 的院內超前部署規則）：】
{CORE_RULES}

【系統為您精準調閱的『{selected_dept}』雙份規範文件內容（內含輔系科目與各年級完整課程規劃）：】
{extracted_bundle}

請根據上述提供的具體規範與雙份文件，進行全方位的交叉比對與修課安排：
1. 畢業學分組成結構：精確說明該生如何滿足 128 總學分。
2. 【核心排課優化重點】學院選修 50 學分的精準開課對齊：
   - 請明確區分出哪【20學分】是精準對齊【文件 A：輔系表】所列出的必修科目。
   - 請深入翻閱【文件 B：課程規劃表】，挑選出那些『沒有寫在輔系表、但屬於該系大一到大四的重要專業課程』，並強烈建議學生利用空檔修讀這些課，以此來填滿剩下的【30學分】學院選修！必須具體列出這些課被安排在哪些年級與學期。
3. 四年修課路徑發展建議：具體列出大一至大四各學年的修課策略表。請將不分系專業必修(25學分)、「{selected_module}」核心課(12學分)、特定系的20學分輔系課、以及從課程規劃表超前部署挑選出的30學分系專業課，完美且合乎邏輯地交織融入各年級的排課中（注意先修課程的邏輯順序，例如統計學要在作業研究前面）。
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
