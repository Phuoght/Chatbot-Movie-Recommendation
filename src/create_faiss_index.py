import faiss
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import normalize


def create_embeddings_for_column(column_data):
    inputs = tokenizer(list(column_data), return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')  # Mean pooling
    return normalize(embeddings)

def create_faiss_index(df, embeddings):
    d = embeddings.shape[1]  # Chiều của vector
    nlist = np.sqrt(len(df))  # Số lượng cụm (clusters)
    quantizer = faiss.IndexFlatL2(d)  # Quantizer để phân cụm
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # Huấn luyện index
    index.train(embeddings)

    # Thêm vector vào index
    index.add(embeddings)

    # Thiết lập số lượng các cụm sẽ kiểm tra
    index.nprobe = min(10, nlist)

    return index

if __name__ == '__main__':
    df_movies = pd.read_csv('../data/data-film-final.csv')
    
    df_movies['title'] = df_movies['title'].apply(lambda x: x.strip().lower())
    df_movies['director'] = df_movies['director'].apply(lambda x: x.strip().lower())
    df_movies['genre'] = df_movies['genre'].apply(lambda x: x.strip().lower())
    df_movies['actor'] = df_movies['actor'].apply(lambda x: str(x).strip().lower())
    df_movies['describe'] = df_movies['describe'].apply(lambda x: x.strip().lower())
    df_movies['country'] = df_movies['country'].apply(lambda x: x.strip().lower())

    movies = df_movies.to_dict('records')

    movie_texts = []

    for movie in movies:
        text = (f"{movie['title']} - {movie['describe']} - {movie['genre']}) "
                f"{movie['director']} - {movie['actor']} "
                f"{movie['country']} - rating: {movie['rating']} - Năm: {movie['release_year']}")
        movie_texts.append(text)

    tokenizer = AutoTokenizer.from_pretrained('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    model = AutoModel.from_pretrained('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')

    # Tokenize đầu vào
    inputs = tokenizer(movie_texts, return_tensors="pt", padding=True, truncation=True)
    # Tạo embedding từ mô hình
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings_all = outputs.last_hidden_state.mean(dim=1).numpy().astype('float32')  # Mean pooling
    embeddings_all = normalize(embeddings_all)

    # Tạo embedding cho director
    director_embeddings = create_embeddings_for_column(df_movies['director'])
    # Tạo embedding cho actor
    df_movies["actor_list"] = df_movies["actor"].dropna().apply(lambda x: x.split(", ") if isinstance(x, str) else [])
    df = df_movies.explode("actor_list").reset_index(drop=True)
    actor_embeddings = create_embeddings_for_column(df['actor_list'])

    index_all = create_faiss_index(df_movies, embeddings_all)
    index_director = create_faiss_index(df_movies,director_embeddings)
    index_actor = create_faiss_index(df, actor_embeddings)

        # Lưu index để sử dụng sau này
    faiss.write_index(index_all, '../data/faiss_index_film.bin')
    faiss.write_index(index_actor, '../data/faiss_index_actor.bin')
    faiss.write_index(index_director, '../data/faiss_index_director.bin')



