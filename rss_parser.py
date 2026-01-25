import feedparser
from bs4 import BeautifulSoup
import re
import datetime
import time
import os
import requests
import json
import io
import pypdf
import traceback
from google import genai  # ✅ 변경된 import

# --- API 키 설정 ---
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# --- Gemini Client 설정 (신규 SDK) ---
gemini_client = None
if GEMINI_API_KEY:
    try:
        # ✅ 신규 SDK 초기화 방식
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini Client 설정 완료")
    except Exception as e:
        print(f"🚨 Gemini Client 설정 실패: {e}")

# --- 사용자 관심사 정의 ---
USER_INTERESTS = """
- Embedding Model, Reranker
- LLM, LLM Quantization
- LoRA 튜닝, Domain-Adaptation, Continual-Learning
- Sparse/Dense Vector, Vector DB & Search
- Retrieval-Augmented Generation (RAG)
"""

def get_paper_relevance_scores_openrouter(papers_batch):
    """
    OpenRouter API (Reasoning) - 기존 로직 유지
    """
    if not papers_batch: return []
    if not OPENROUTER_API_KEY:
        print("🚨 OpenRouter API Key가 없습니다.")
        return []

    prompt_papers_section = ""
    for i, paper in enumerate(papers_batch):
        title = re.sub(r'\s*\(v\d+\)$', '', paper.title)
        summary = BeautifulSoup(paper.summary, 'html.parser').get_text(separator=" ", strip=True)
        prompt_papers_section += f'\n{{ "id": {i}, "title": "{title}", "abstract": "{summary[:900]}" }}'

    system_prompt = "You are an expert AI researcher. Analyze the papers carefully based on the user's interests."
    
    user_prompt = f"""
    Evaluate the relevance of the following papers based on my interests.
    
    --- My Interests ---
    {USER_INTERESTS}
    --------------------

    **Reasoning Task:**
    1. Think step-by-step about how each paper's abstract matches the specific technical keywords in my interests.
    2. Assign a relevance score from 0 to 100.
    3. Exclude papers focusing on specific languages (Thai, Arabic, etc.) unless it's Korean.

    --- Papers to Evaluate ---
    [
        {prompt_papers_section}
    ]
    --------------------

    **Output Format:**
    Provide the output ONLY as a valid JSON list of objects. Do not include markdown code blocks.
    Example: [ {{"id": 0, "score": 15}}, {{"id": 1, "score": 95}} ]
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "nvidia/nemotron-3-nano-30b-a3b:free", 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        # "reasoning": {"enabled": True}, 
        # "temperature": 0.2 
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=60
        )
        response.raise_for_status()
        
        result_json = response.json()
        choice = result_json['choices'][0]
        content_text = choice['message']['content']

        clean_json_text = content_text.replace("```json", "").replace("```", "").strip()
        scores_data = json.loads(clean_json_text)

        scored_papers = []
        scores_map = {item['id']: item['score'] for item in scores_data}

        for i, paper in enumerate(papers_batch):
            score = scores_map.get(i, 0)
            if score >= 75: 
                scored_papers.append({'paper': paper, 'score': score})
        
        print(f"   - OpenRouter 분석 완료: {len(papers_batch)}개 중 {len(scored_papers)}개 선정.")
        return scored_papers

    except Exception as e:
        print(f"   - ❗ OpenRouter API 호출/파싱 에러: {e}")
        traceback.print_exc()
        return []

def summarize_paper_gemini(paper_url):
    """
    ✅ Gemini 신규 SDK 적용 (google-genai)
    """
    if not gemini_client: return "Gemini Client 미설정으로 요약 불가"
    
    try:
        pdf_url = paper_url.replace('/abs/', '/pdf/')
        res = requests.get(pdf_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
        pdf_file = io.BytesIO(res.content)
        reader = pypdf.PdfReader(pdf_file)
        # 앞 9페이지 추출
        text = "".join([page.extract_text() or "" for page in reader.pages[:9]])

        # 간단한 텍스트 전처리
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if not cleaned_text:
            return "PDF에서 텍스트를 추출할 수 없습니다."
        
        prompt = f"""
        You are a research assistant. Read the following excerpt from a research paper and provide a structured summary in KOREAN. 
        The summary must follow this structure:
        - **1. 3-line Summary**
        - **2. Problem Statement**
        - **3. Proposed Method**
        - **4. Key Contribution**
        - **5. Results & Evaluation**
        
        --- Paper Excerpt ---
        Text: {cleaned_text[:12000]} 
        --- End of Excerpt ---

        Your structured summary in Korean:
        """
        
        # ✅ 변경된 호출 방식: client.models.generate_content
        # 모델명은 'gemini-2.0-flash' (최신) 혹은 'gemini-1.5-flash' 사용 권장
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        
        return response.text.strip()
        
    except Exception as e:
        return f"요약 실패: {e}"

def send_discord_briefing(papers_list, category_name):
    if not DISCORD_WEBHOOK_URL: return
    
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    header = {"content": f"## 🧠 {today_str} **{category_name}** 논문 (Reasoning Filtered)"}
    requests.post(DISCORD_WEBHOOK_URL, json=header)

    for item in papers_list:
        paper = item['paper']
        embed = {
            "title": f"📄 {paper.title[:200]}",
            "url": paper.link,
            "description": item.get('summary', '요약 없음')[:2000],
            "color": 3447003, 
            "fields": [
                {"name": "Score", "value": f"**{item['score']}** / 100", "inline": True},
                {"name": "Published", "value": paper.published.split('T')[0], "inline": True}
            ]
        }
        requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]})
        time.sleep(1)

# --- 메인 실행 로직 ---
arxiv_urls = ["http://export.arxiv.org/rss/cs.AI", "http://export.arxiv.org/rss/cs.CL"]
yesterday = datetime.date.today() - datetime.timedelta(days=1)

print(f"📅 기준 날짜: {yesterday}")

for url in arxiv_urls:
    category = url.split('/')[-1]
    print(f"\n[{category}] 수집 시작...")
    
    feed = feedparser.parse(url)
    recent_papers = [
        e for e in feed.entries 
        if datetime.date(e.published_parsed.tm_year, e.published_parsed.tm_mon, e.published_parsed.tm_mday) >= yesterday
    ]

    if not recent_papers:
        print(" -> 새 논문 없음.")
        continue

    # 배치 처리 (OpenRouter Reasoning 사용)
    top_papers = []
    batch_size = 8 
    for i in range(0, len(recent_papers), batch_size):
        batch = recent_papers[i:i+batch_size]
        print(f" -> 배치 {i//batch_size + 1} 평가 중 (OpenRouter)...")
        scores = get_paper_relevance_scores_openrouter(batch)
        top_papers.extend(scores)
        time.sleep(20) 
    
    if len(top_papers) == 0:
        raise ValueError('평가된 논문이 존재하지 않습니다. 평가 단계 및 API를 확인하세요.')

    # 상위 8개 선정 및 요약
    top_papers.sort(key=lambda x: x['score'], reverse=True)
    final_list = top_papers[:8]

    for item in final_list:
        print(f" -> 요약 생성 중: {item['paper'].title[:30]}...")
        item['summary'] = summarize_paper_gemini(item['paper'].link)
        time.sleep(20)

    if final_list:
        send_discord_briefing(final_list, category)

print("\n완료.")