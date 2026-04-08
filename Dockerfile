# 빌드를 가볍게 하기 위해 python slim 이미지를 사용합니다.
FROM python:3.10-slim

# 컨테이너 내 작업 디렉토리 설정
WORKDIR /app

# python이 pyc 파일을 쓰지 않도록설정 (성능 및 용량 최적화)
ENV PYTHONDONTWRITEBYTECODE 1
# python 결과물을 버퍼링 없이 즉시 출력하도록 설정
ENV PYTHONUNBUFFERED 1

# 시스템 필수 패키지 설치 및 캐시 제거 (이미지 최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 가져오기 (레이어 캐싱을 위해 코드 전체 복사보다 먼저 수행)
COPY requirements.txt .

# pip 업데이트 및 패키지 설치
# 팁: 이미지 크기 최적화를 위해 PyTorch CPU 버전을 명시적으로 우선 설치할 수 있습니다.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 전체 소스 코드 복사
COPY . .

# 컨테이너가 8000 포트에서 수신하도록 설정
EXPOSE 8000

# 컨테이너 시작 시 uvicorn 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
