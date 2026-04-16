import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- GIẢ LẬP DỮ LIỆU MẪU ---
data = {
    'gia': [5000, 7000, -100, 6500, 200000, 7000], # có giá âm và outlier cực lớn
    'so_phong': [2, 3, 0, np.nan, 5, 3],           # có giá trị thiếu và bằng 0
    'quan': ['Quận 1', 'Quận 3', 'Quận 1', 'Quận 3', 'Quận 2', 'Quận 3'],
    'mo_ta': [
        'Nhà đẹp hẻm xe hơi Quận 1',
        'Cần bán gấp nhà lầu Quận 3',
        'Nhà nát giá rẻ',
        'Căn hộ chung cư cao cấp',
        'Biệt thự sân vườn rộng',
        'Cần bán gấp nhà lầu Quận 3' # Tin trùng lặp
    ]
}
df = pd.DataFrame(data)

# --- BƯỚC 2: XỬ LÝ DỮ LIỆU BẨN ---
# 1. Loại bỏ dữ liệu vô lý (giá âm, số phòng = 0)
df = df[(df['gia'] > 0) & (df['so_phong'] != 0)]

# 2. Điền missing values (Dùng Median cho số phòng)
imputer = SimpleImputer(strategy='median')
df['so_phong'] = imputer.fit_transform(df[['so_phong']])

# 3. Loại bỏ duplicate records (trùng hoàn toàn các cột)
df = df.drop_duplicates().reset_index(drop=True)


# --- BƯỚC 3: XỬ LÝ OUTLIERS (IQR) ---
def handle_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    # Capping: Giới hạn giá trị trong khoảng cho phép
    df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

df = handle_outliers_iqr(df, 'gia')


# --- BƯỚC 4: CHUẨN HÓA & BIẾN ĐỔI ---
# 1. Scaling Numerical (Min-Max)
scaler = MinMaxScaler()
df[['gia_scaled', 'so_phong_scaled']] = scaler.fit_transform(df[['gia', 'so_phong']])

# 2. One-hot Encoding cho cột 'quan'
encoder = OneHotEncoder(sparse_output=False)
quan_encoded = encoder.fit_transform(df[['quan']])
quan_df = pd.DataFrame(quan_encoded, columns=encoder.get_feature_names_out(['quan']))
df = pd.concat([df, quan_df], axis=1)


# --- BƯỚC 5: TEXT SIMILARITY (PHÁT HIỆN TRÙNG LẶP NỘI DUNG) ---
# 1. TF-IDF biến mô tả thành vector
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['mo_ta'])

# 2. Tính Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# 3. Tìm các tin có độ trùng lặp > 80% (không tính chính nó)
duplicates = []
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > 0.8:
            duplicates.append((i, j, cosine_sim[i, j]))

print("--- KẾT QUẢ ---")
print(df[['mo_ta', 'gia_scaled', 'quan_Quận 1']])
print(f"\nCác cặp tin trùng lặp nội dung: {duplicates}")