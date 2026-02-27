import cv2
import numpy as np
import os
import random
import yaml
import shutil

# 1. 讀取設定檔
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

NUM_IMAGES = config['dataset']['num_images']
BASE_DIR = config['dataset']['output_dir']
TRAIN_RATIO = config['dataset']['train_ratio']

TEXT_VOCAB = ["Amplitude", "Time", "ms", "sec", "cycle", "Hz", "Delay", "Signal", "100", "20", "Rise", "Fall"]

def setup_directories():
    """建立標準 YOLO Train/Val 資料夾結構"""
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(BASE_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, split, 'labels'), exist_ok=True)

def generate_dataset():
    """主生成邏輯"""
    setup_directories()
    
    for i in range(NUM_IMAGES):
        # 決定這張圖要放在 Train 還是 Val
        split = 'train' if random.random() < TRAIN_RATIO else 'val'
        
        # 讀取畫布設定
        img_w = random.randint(config['canvas']['width_min'], config['canvas']['width_max'])
        img_h = random.randint(config['canvas']['height_min'], config['canvas']['height_max'])
        img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
        
        # 加入干擾文字 (負樣本)
        for _ in range(random.randint(3, 8)):
            text = random.choice(TEXT_VOCAB)
            tx, ty = random.randint(10, img_w - 100), random.randint(20, img_h - 20)
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.4, 0.8), (50,50,50), 1)

        labels = []
        num_arrows = random.randint(config['arrows']['min_count'], config['arrows']['max_count'])
        
        for _ in range(num_arrows):
            arrow_w = random.randint(50, img_w // 2)
            x1 = random.randint(20, img_w - arrow_w - 20)
            x2 = x1 + arrow_w
            y = random.randint(30, img_h - 30)
            head_size = random.randint(10, 25)
            
            # 隨機顏色與畫箭頭的主線
            color = (0, 0, 0)
            if config['arrows']['color_variety'] and random.random() > 0.8:
                color = (180, 50, 120) # 隨機紫色干擾
                
            cv2.line(img, (x1, y), (x2, y), color, random.randint(1, 3))
            
            # 畫箭頭三角形 (簡化版實心箭頭)
            pts_left = np.array([[x1, y], [x1+head_size, y-head_size//2], [x1+head_size, y+head_size//2]], np.int32)
            cv2.fillPoly(img, [pts_left], color)
            pts_right = np.array([[x2, y], [x2-head_size, y-head_size//2], [x2-head_size, y+head_size//2]], np.int32)
            cv2.fillPoly(img, [pts_right], color)
            
            # YOLO Bounding Box 座標計算
            bbox_x_center, bbox_y_center = (x1 + x2) / 2.0 / img_w, y / img_h
            bbox_width, bbox_height = (x2 - x1) / img_w, (head_size * 1.5) / img_h
            labels.append(f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # 存檔
        base_name = f"wave_{i:04d}"
        cv2.imwrite(os.path.join(BASE_DIR, split, 'images', f"{base_name}.jpg"), img)
        with open(os.path.join(BASE_DIR, split, 'labels', f"{base_name}.txt"), 'w') as f:
            f.write("\n".join(labels) + "\n")

    print(f"✅ 成功生成並切分 {NUM_IMAGES} 張資料至 {BASE_DIR}/ 目錄下！")

if __name__ == "__main__":
    generate_dataset()