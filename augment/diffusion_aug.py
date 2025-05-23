from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import random, torch
from PIL import Image
from pathlib import Path

# 더미 함수입니다. 실제로는 OpenPose keypoint 추출 함수로 대체하세요.
def extract_openpose(img_path):
    return Image.open(img_path).convert("RGB")

class DiffusionAugmentDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, prob=0.3, cache_dir="cache/diff", 
                 sd_model="runwayml/stable-diffusion-v1-5", controlnet="lllyasviel/control_v11p_sd15_openpose"):
        self.base = base_ds              # 기존 ImageDataset 인스턴스
        self.prob = prob
        self.cache = cache_dir
        self.sd_model = sd_model
        self.controlnet = controlnet
        self.pipe = self._build_pipe()
    
    def _build_pipe(self):
        cn = ControlNetModel.from_pretrained(self.controlnet)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.sd_model, controlnet=cn, safety_checker=None
        ).to("cuda").half()
        pipe.enable_xformers_memory_efficient_attention()
        return pipe
    
    def _generate(self, img_path):
        fn = Path(img_path).stem + "_aug.png"
        save_p = Path(self.cache) / fn
        if save_p.exists():
            return Image.open(save_p)
        pose = extract_openpose(img_path)
        out = self.pipe(
            prompt="full-body photo",
            image=Image.open(img_path),
            control_image=pose, num_inference_steps=20
        ).images[0]
        save_p.parent.mkdir(parents=True, exist_ok=True)
        out.save(save_p)
        return out
    
    def __getitem__(self, idx):
        # base dataset는 (img, pid, camid, cloth)를 반환한다고 가정합니다.
        img, pid, camid, cloth = self.base[idx]
        # base dataset 내부에 원본 이미지 경로 정보가 있다면 사용합니다.
        # 여기서는 self.base.dataset[idx][0]이 이미지 경로라고 가정합니다.
        if random.random() < self.prob:
            img = self._generate(self.base.dataset[idx][0])
        return self.base.transform(img), pid, camid, cloth
    
    def __len__(self):
        return len(self.base)