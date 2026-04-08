import io
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# MLOps 파이프라인의 시작점: 가벼운 모델(MobileNetV2) 로딩
# 실제 환경에서는 로컬에 저장된 가중치(.pth, .onnx)나 모델 레지스트리(MLflow, S3 등)에서 모델을 불러오게 됩니다.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval() # 추론 모드 설정

# 예시용 더미 클래스 매핑 (실제 학습된 모델 클래스에 맞게 수정 필요)
RECYCLE_CLASSES = {
    0: "Plastic (플라스틱)",
    1: "Paper (종이)",
    2: "Glass (유리)",
    3: "Metal (금속/캔)",
    4: "General Waste (일반 쓰레기)"
}

def transform_image(image_bytes):
    """
    이미지 바이트를 입력받아 MobileNet 모델 입력 크기(224x224) 및 정규화를 수행하는 전처리 함수.
    """
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return my_transforms(image).unsqueeze(0)

def predict_image(image_bytes):
    """
    전처리된 이미지를 모델에 통과시켜 클래스와 신뢰도를 반환합니다.
    """
    tensor = transform_image(image_bytes)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # 모델의 예측 결과 (인덱스)
        confidence, y_hat = probabilities.max(0)
        class_idx = y_hat.item()
        
        # 예시 데모를 위해 임의로 5개의 재활용 클래스로 모듈러 연산 매핑 (실제 사용 시 제거)
        mapped_idx = class_idx % 5 
        
    return RECYCLE_CLASSES.get(mapped_idx, "Unknown"), confidence.item() * 100
