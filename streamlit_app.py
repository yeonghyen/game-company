#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '10cy4AJ9d-gamTdZ8dzyeFoo6VKLBwtVq'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/qxr98Rx/d5886780.jpg",
            "https://i.ibb.co/S5D40BB/thumbnail-home.jpg",
            "https://i.ibb.co/Rb9XYkg/3-F3-F-3-F3-F3-F.webp"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=HNCKjStDQUU",
            "https://www.youtube.com/watch?v=LOamz1pMCyA",
            "https://www.youtube.com/watch?v=PMv7enz-5tM"
        ],
        'texts': [
            "블리자드 오버워치2 14시즌 시작(with 해저드)",
            "블리자드 하스스톤를 드라마로 만든 하트스톤 예고편",
            "블리자드 스타크래프트 임진록(임요환과 홍진호) 매치 경기(2년전)"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/HKtRz5n/og-image.png",
            "https://i.ibb.co/xGt7dqt/nexon-logo.png",
            "https://i.ibb.co/GvYntnz/images.jpg"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=N9Es4p1ug0U",
            "https://www.youtube.com/watch?v=LSrjRi2-1ik",
            "https://www.youtube.com/watch?v=O3tGuW5ykZk"
        ],
        'texts': [
            "넥슨 30주년",
            "넥슨 피파 쵸단 광고",
            "넥슨 메이플스토리 피아노 연주(연주자:유후)"
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/VQz1Fg0/images.jpg",
            "https://i.ibb.co/L6Bqt4C/3a458fb831e85841a62acde2913927a5.jpg",
            "https://i.ibb.co/1Yg7W6c/66561-96458-724.png"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=tm9czVwjelA&t=7s",
            "https://www.youtube.com/watch?v=AQXU0plQEBA",
            "https://www.youtube.com/watch?v=6CGWWYrjXvY"
        ],
        'texts': [
            "라이엇 게임즈 롤에서 t1과 페이커 선수의 올해의 이스포츠팀/올해의 선수상",
            "라이엇 게임즈 롤에서 만든 애니인 아케",
            "라이엇 게임즈 발로란트에서 이벤트 진행(케미폭VAL)"
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

