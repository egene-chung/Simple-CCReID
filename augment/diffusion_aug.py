from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import random, torch
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# MediaPipe 포즈 추출 함수
def extract_openpose(img_path):
    # 이미지 로드
    image = cv2.imread(img_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {img_path}")
        # 더미 이미지 반환
        return Image.open(img_path).convert("RGB")
    
    # RGB로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Pose 모델 초기화
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # 검정 배경 생성
    height, width = image.shape[:2]
    pose_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5
    ) as pose:
        # 포즈 감지
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # 포즈 랜드마크 그리기
            mp_drawing.draw_landmarks(
                pose_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # OpenPose 스타일로 변환 (흰색 선과 조인트)
            pose_image = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
            
            # 디버깅을 위해 저장
            debug_path = Path(f"cache/debug/{Path(img_path).stem}_pose.png")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), cv2.cvtColor(pose_image, cv2.COLOR_RGB2BGR))
            
            return Image.fromarray(pose_image)
        else:
            print(f"포즈를 감지할 수 없습니다: {img_path}")
            # 포즈가 감지되지 않으면 원본 이미지 반환
            return Image.open(img_path).convert("RGB")

# diffusion callback 정의
def diffusion_callback(step: int, timestep: int, latents: torch.Tensor):
    print(f"[Diffusion Callback] Step: {step} | Timestep: {timestep}")
    
class DiffusionAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, config):
        self.base = base_ds
        self.prob = config.DIFFUSION_AUG.PROB
        self.cache = "cache/diff"
        self.sd_model = config.DIFFUSION_AUG.SD_MODEL
        self.controlnet = config.DIFFUSION_AUG.CONTROLNET
        self.pipe = self._build_pipe()
        print("DiffusionAugmentDataset 초기화 완료")
    
    def _build_pipe(self):
        print(f"모델 로드 시작: {self.sd_model}, {self.controlnet}")
        cn = ControlNetModel.from_pretrained(self.controlnet)
        
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model, controlnet=cn, safety_checker=None
        ).to("cuda")
        
        pipe.enable_attention_slicing()
        
        print("Pipeline 생성 완료")
        return pipe
    
    def _generate(self, img_path):
        fn = Path(img_path).stem + "_aug.png"
        save_p = Path(self.cache) / fn
        print(f"Save path: {save_p}")
        if save_p.exists():
            print("이미 캐시된 이미지가 존재합니다.")
            try:
                return Image.open(save_p).convert("RGB")
            except Exception as e:
                print(f"캐시 이미지 로드 오류: {str(e)}")
                # 캐시 이미지 로드 실패 시 원본 이미지 반환
                return Image.open(img_path).convert("RGB")
        
        try:
            # MediaPipe를 사용하여 포즈 추출
            pose = extract_openpose(img_path)
            
            # 원본 이미지 로드
            orig_img = Image.open(img_path).convert("RGB")
            
            # 메모리 관리 및 안정성을 위해 inference_steps 감소
            with torch.no_grad():  # 메모리 효율성 향상
                out = self.pipe(
                    prompt="same person wearing different clothes, maintain identity, same face same pose",
                    negative_prompt="different person, different face, different identity",
                    image=orig_img,
                    control_image=pose,
                    num_inference_steps=15,  # 20에서 15로 감소
                    # callback 제거하여 안정성 향상
                ).images[0]
            
            # 메모리 정리
            torch.cuda.empty_cache()
            
            print(f"Creating directory: {save_p.parent}")
            save_p.parent.mkdir(parents=True, exist_ok=True)
            out.save(save_p)
            print(f"Image saved: {save_p}")
            return out
        except Exception as e:
            print(f"이미지 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 이미지 반환
            return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx):
        # 원본 데이터셋에서 이미지 경로와 메타데이터 가져오기
        img_path, pid, camid, cloth = self.base.dataset[idx]
        
        # diffusion 증강 적용 여부 결정
        if random.random() < self.prob:
            try:
                # diffusion 모델로 이미지 생성
                generated_img = self._generate(img_path)
                # 생성된 이미지에 transform 적용
                img_tensor = self.base.transform(generated_img)
            except Exception as e:
                print(f"이미지 생성 오류: {str(e)}")
                # 오류 발생 시 원본 이미지 로드하여 transform 적용
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.base.transform(img)
        else:
            # diffusion을 적용하지 않는 경우 원본 이미지 로드하여 transform 적용
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.base.transform(img)
        
        return img_tensor, pid, camid, cloth
    
    def __len__(self):
        return len(self.base)