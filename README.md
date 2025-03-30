# Chatbot Gợi Ý Phim

## Giới thiệu
Chatbot này giúp người dùng **tìm kiếm và gợi ý phim** theo sở thích bằng cách:
- **Crawl dữ liệu phim** từ **phimmoichill.best** bằng Scrapy.
- **Embedding dữ liệu phim & câu hỏi** bằng **Sup-SimCSE-VietNamese-Phobert-Base**.
- **Tạo FAISS Index** để tìm kiếm phim nhanh chóng.
- **Sử dụng DeepSeek AI** để trả lời dựa trên phim đã tìm được.

*Lưu ý:* AI **chỉ gợi ý phim từ dữ liệu đã có**, **không tự tạo phim mới**.

---


## Công nghệ sử dụng
- **Web Scraping**: Scrapy  
- **NLP & Embedding**: `Sup-SimCSE-VietNamese-Phobert-Base`  
- **Tìm kiếm nhanh**: FAISS  
- **AI Model**: DeepSeek  
- **Giao diện**: Flask hoặc giao diện CLI/GUI đơn giản  


## Cách chạy chatbot
- Clone repository: git clone https://github.com/Phuoght/Chatbot-Movie-Recommendation.git
- cd Chatbot-Movie-Recommendation
- Tải file: git lfs pull

### Cài đặt thư viện cần thiết
- pip install -r requirements.txt

### Chạy chatbot

#### API Flask
python app.py
Truy cập **http://127.0.0.1:5000/** 


## Cách hoạt động
1. **Người dùng nhập câu hỏi** về phim mong muốn.
2. **Embedding câu hỏi** bằng **Sup-SimCSE-VietNamese-Phobert-Base**.
3. **FAISS tìm kiếm phim phù hợp** dựa trên dữ liệu đã index.
4. **DeepSeek AI generate câu trả lời** dựa trên danh sách phim tìm được.
5. **Chatbot phản hồi gợi ý phim**.

## 🔧 Môi trường `.env`
Bạn cần tạo file `.env` trong thư mục chính để lưu thông tin quan trọng như API key:
- DEEPSEEK_API_KEY=your_api_key_here

## Hiệu suất chatbot
- **Tốc độ phản hồi**: ~10 giây (tìm kiếm nhanh nhưng call api phản hồi lâu)  
- **Độ chính xác gợi ý**: ~80% dựa trên phản hồi thử nghiệm.

##  Định hướng phát triển
- Cải thiện hiệu suất phản hồi chatbot.
- Cải thiện thuật toán để tìm kiếm chính xác hơn.
- Tích hợp chatbot vào Messenger/Zalođể tiện lợi hơn.


