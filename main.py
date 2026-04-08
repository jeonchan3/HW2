from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from ml_model import predict_image

app = FastAPI(
    title="Recycle Sorting API",
    description="재활용 쓰레기(플라스틱, 종이, 유리 등) 분류를 위한 이미지 인식 API 서버입니다.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Recycle Sorting MLOps API Server is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 1. 파일 확장자 및 타입 검사
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    
    try:
        # 2. 업로드된 파일 읽기
        image_bytes = await file.read()
        
        # 3. 모델을 사용한 추론 호출
        predicted_class, confidence = predict_image(image_bytes)
        
        # 4. 결과 반환
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 로컬 개발용 서버 실행 (실제 배포 시에는 uvicorn CLI 명령어로 실행하는 것이 좋습니다)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
