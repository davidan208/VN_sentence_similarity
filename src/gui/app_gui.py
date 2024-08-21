import streamlit as st
import requests
import time

st.title("ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG NGỮ NGHĨA CỦA HAI CÂU TIẾNG VIỆT.")

BACKEND_URL = "http://localhost:8000"

st.sidebar.title("Người thực hiện:")
st.sidebar.write("Nguyễn Đình Thiên Ân - 23C15019")

method = st.sidebar.selectbox(
    "Chọn phương pháp",
    ("phobert-mean", "phobert-cls", "sup-simcse", "multi-miniLM", "vi-sbert")
)

task = st.sidebar.radio(
    "Nhiệm vụ",
    ("Đánh giá độ tương đồng", "Tìm câu tương đồng nhất trong đoạn văn")
)

if task == "Đánh giá độ tương đồng":
    st.header("Đánh giá độ tương đồng")
    
    sentence1 = st.text_input("Nhập câu thứ nhất:")
    sentence2 = st.text_input("Nhập câu thứ hai:")

    if st.button("Đánh giá độ tương đồng"):
        if sentence1 and sentence2:
            data = {
                "sentence1": sentence1,
                "sentence2": sentence2
            }
            start = time.time()
            response = requests.post(f"{BACKEND_URL}/similarity?method={method}", json=data)
            end = time.time() - start
            if response.status_code == 200:
                result = response.json()
                st.success(f"Độ tương đồng: {result['similarity']:.4f}")
                st.info(f"Thời gian thực thi: {end:.4f} giây")
            else:
                st.error(f"Error: {response.text}")
        else:
            st.warning("Hãy nhập cả hai câu.")

    st.markdown("***Lưu ý 1:** Chương trình chưa có chức năng tự động sửa lỗi chính tả hay ngữ pháp, vui lòng nhập câu đúng chính tả, đúng ngữ pháp.*")
    st.markdown("***Lưu ý 2:** Chương trình chạy bằng CPU, nên các mô hình đều nhận tối đa 256 token đã được truncation, phần dư sẽ bị bỏ qua để tiết kiệm thời gian.*")

elif task == "Tìm câu tương đồng nhất trong đoạn văn":
    st.header("Tìm câu tương đồng nhất trong đoạn văn")
    
    sentence = st.text_input("Nhập câu:")
    paragraph = st.text_area("Nhập đoạn văn:")

    if st.button("Thực hiện tìm"):
        if sentence and paragraph:
            data = {
                "sentence": sentence,
                "paragraph": paragraph
            }
            start = time.time()
            response = requests.post(f"{BACKEND_URL}/most-similar?method={method}", json=data)
            end = time.time() - start
            if response.status_code == 200:
                result = response.json()
                st.success(f"Câu có độ tương đồng cao nhất: {result['most_similar_sentence']} - {result['score']}")
                st.info(f"Thời gian thực thi: {end:.4f} giây")
            else:
                st.error(f"Error: {response.text}")
        else:
            st.warning("Hãy nhập đầy đủ câu và đoạn văn.")

    st.markdown("***Lưu ý 1:** Chương trình chưa có chức năng tự động sửa lỗi chính tả hay ngữ pháp, vui lòng nhập câu đúng chính tả, đúng ngữ pháp.*")
    st.markdown("***Lưu ý 2:** Chương trình chạy bằng CPU, nên các mô hình đều nhận tối đa 256 token đã được truncation, phần dư sẽ bị bỏ qua để tiết kiệm thời gian.*")

# Display information about the selected method
st.sidebar.markdown("---")
st.sidebar.subheader("Thông tin mô hình")
method_info = {
    "phobert-mean": "Dùng mô hình Phobert-base-v2 lấy trung bình token output.",
    "phobert-cls": "Dùng mô hình Phobert-base-v2 lấy output token CLS.",
    "sup-simcse": "Dùng mô hình simCSE huấn luyện có giám sát.",
    "multi-miniLM": "Dùng mô hình paraphrase-multilingual-MiniLM-L12-v2.",
    "vi-sbert": "Dùng mô hình sbert.",
}
st.sidebar.info(method_info[method])
