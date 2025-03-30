from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import faiss
import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import asyncio
from sklearn.preprocessing import normalize
import re


app = Flask(__name__)

# Load biến môi trường
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load FAISS index và dữ liệu

df_movies = pd.read_csv('data/data-film-final.csv')
faiss_index_all = faiss.read_index('data/faiss_index_film.bin')
faiss_index_director = faiss.read_index('data/faiss_index_director.bin')
faiss_index_actor = faiss.read_index('data/faiss_index_actor.bin')

# Load model NLP
tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
model = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

# Tạo Client API
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

suggestion_keywords = [ 
                        "thêm gợi ý", "có phim nào tương tự", "có phim nào giống vậy",
                        "có phim nào cùng thể loại", "còn phim nào khác", "gợi ý thêm",
                        "có lựa chọn nào khác", "phim tương tự", "còn gì nữa không",
                        "gợi ý phim khác", "có bộ nào tương tự", "còn phim nào hay không",
                        "thêm vài phim đi", "phim nào giống", "phim tương đương", "thêm", "tiếp",
                        "tương tự vậy còn gì không", "còn phim nào như thế không", "thêm nữa đi",
                        "có phim nào gần giống không", "có bộ phim nào liên quan không",
                        "gợi ý tiếp", "có bộ nào nổi bật không", "tiếp tục gợi ý",
                        "còn thể loại nào khác không", "còn phim nào đáng xem không",
                        "đề xuất thêm", "còn gì để xem không", "gợi ý một phim khác",
                        "thêm danh sách phim đi", "phim nào hợp với cái này", "phim nào phong cách giống vậy",
                        "có series nào như thế không", "phim nào hợp gu này", "có gì đặc sắc hơn không",
                        "phim nào có vibe tương tự", "có phim nào khác thú vị không", "còn bộ nào hay ho không",
                        "thêm vài cái lựa chọn nữa đi", "phim nào đáng xem hơn", "thêm đề xuất",
                        "có gì hot hơn không", "phim nào chất lượng hơn", "có cái nào độc đáo hơn không"
                        ]

meaningful = ["phim", "thể loại", "chủ đề", "gợi ý", "thích xem", "hay", "đề xuất", "xem", "nào"]

df_movies['actor'] = df_movies['actor'].astype(str).str.strip().str.lower()
df_movies['director'] = df_movies['director'].astype(str).str.strip().str.lower()

df_movies["actor_list"] = df_movies["actor"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else [])
df_for_actor = df_movies.explode("actor_list").reset_index(drop=True)

actors = set(df_for_actor["actor_list"].explode().dropna())
actors = {actor for actor in actors if len(actor) > 5}

directors = set(df_movies['director'].dropna().astype(str).tolist())
directors = {director for director in directors if len(director) > 5}

class ConversationMemory:
    def __init__(self):
        self.history = []

    def add_message(self, message):
        if any(keyword in message for keyword in suggestion_keywords):
            self.history.append(message)
        else:
            self.history = [message]  # Reset và chỉ giữ câu mới nhất

    def get_context(self):
        return " - ".join(self.history) if self.history else ""

memory = ConversationMemory()
add_topk = 5
# Hàm lấy embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return normalize(embedding)

# Hàm tìm kiếm FAISS
def filter_result(faiss_index, top_k, query_vector, df):
    distances, I = faiss_index.search(query_vector, top_k)
    return [df.iloc[idx] for idx in I[0] if 0 <= idx < len(df)]

def search_answers(query, top_k=5):
    try:
        query_vector = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
        if len(memory.history) > 1 :
            top_k = top_k + add_topk * (len(memory.history) - 1)
        if any(str(actor) in query for actor in actors):
            result = filter_result(faiss_index_actor, top_k, query_vector, df_for_actor)
            print(result)
            return result
        elif any(str(director) in query for director in directors):
            result = filter_result(faiss_index_director, top_k, query_vector,df_movies)
            print(result)
            return result
        else:
            result = filter_result(faiss_index_all, top_k, query_vector, df_movies)
            print(result)
            return result
            
    except Exception as e:
        print(f"Lỗi tìm kiếm: {e}")
        return []

# Hàm gọi DeepSeek API
def call_api(prompt, answers_text):
    try:
        full_prompt = f"""
        Bạn là một chuyên gia tư vấn phim, am hiểu sâu về các thể loại phim, đánh giá chuyên môn và sở thích người xem.
        Ngữ cảnh: Người dùng hỏi về {prompt}.
        Tôi cũng đã tìm được các bộ phim: 
        {answers_text}
        Yêu cầu phản hồi:
        Nếu lớn hơn 5 bộ phim thì hãy bỏ 5 bộ đầu đi chỉ lấy 5 bộ cuối cùng thôi.
        Nếu không phù hợp với yêu cầu người dùng, bỏ qua không lấy phim đó để trình bày, chỉ trình bày những phim có liên quan tới câu hỏi người dùng.
        Bỏ cái lưu ý cuối cùng đi.
        Chỉ sử dụng thông tin từ dữ liệu phim đã được cung cấp. Không sáng tạo thêm nội dung. Hãy cung cấp mô tả chính xác về bộ phim đã được tìm thấy
        Giới hạn: Không trả lời ngoài chủ đề phim ảnh. Luôn đảm bảo thông tin chính xác và hữu ích.
        """
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": full_prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Lỗi API: {e}")
        return "Xin lỗi, có lỗi xảy ra khi tư vấn phim."

async def generate_response(user_query):
    user_query = user_query.strip().lower()
    if len(user_query) < 3:
        return "Câu hỏi quá ngắn, vui lòng nhập rõ ràng hơn."
    if not re.search(r'[a-zA-ZÀ-Ỹà-ỹ]', user_query):
        return "Vui lòng nhập nội dung hợp lệ, làm rõ câu hỏi giúp mình nhé !"
    print(f"User input: {user_query}")  # Debug

    # Kiểm tra xem câu hỏi có liên quan không
    if not (any(keyword in user_query for keyword in meaningful) or any(keyword in user_query for keyword in suggestion_keywords)):
        return "Tôi chưa hiểu bạn muốn gì! Bạn có thể làm rõ hơn không?"

    # Cập nhật bộ nhớ và lấy ngữ cảnh
    memory.add_message(user_query)
    extended_query = memory.get_context() or user_query  # Nếu không có lịch sử, dùng query gốc
    print(f"Extended query: {extended_query}")  # Debug

    # Tìm kiếm phim
    result = await asyncio.to_thread(search_answers, extended_query)
    if not result:
        return "Xin lỗi, không tìm thấy phim phù hợp. Hãy thử từ khóa khác!"

    # Chuyển kết quả thành text
    movies = [m.to_dict() if isinstance(m, pd.Series) else m for m in result]

    answers_text = '\n'.join([f"{m['title']} ({m['release_year']}): {m['describe']} - Đạo diễn: {m['director']} - Diễn viên: {m['actor']}" for m in movies])

    # Gọi DeepSeek để tạo phản hồi
    response = await asyncio.to_thread(call_api, extended_query, answers_text)
    return response

# Flask Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(generate_response(user_input))
    except Exception as e:
        response = f"Xin lỗi, lỗi xảy ra: {e}"
    return jsonify({"user_input": user_input, "bot_response": response})

@app.route("/clear", methods=["POST"])
def clear():
    global memory, flag
    memory = ConversationMemory()
    flag = False

    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)
