import cv2
import numpy as np
import os
import random
import math

# ================= 參數與目錄設定 =================
NUM_IMAGES = 1000 # 生成數量

os.makedirs('dataset_v2/images', exist_ok=True)
os.makedirs('dataset_v2/labels', exist_ok=True)

# 模擬真實時序圖常見的干擾文字
TEXT_VOCAB = ["Amplitude", "Time", "ms", "sec", "cycle", "Hz", "Delay", "Signal", 
              "0", "1", "-1", "100", "20", "Rise", "Fall", "1/10th"]

def draw_solid_triangle_arrow(img, x1, x2, y, color, thickness, head_size):
    """自定義函數：繪製帶有實心三角形或開放式的雙箭頭"""
    # 畫中間的主線條
    cv2.line(img, (x1, y), (x2, y), color, thickness)
    
    style = random.choice(['solid', 'open'])
    
    if style == 'solid':
        # 左箭頭 (◀)
        pts_left = np.array([[x1, y], [x1+head_size, y-head_size//2], [x1+head_size, y+head_size//2]], np.int32)
        cv2.fillPoly(img, [pts_left], color)
        # 右箭頭 (▶)
        pts_right = np.array([[x2, y], [x2-head_size, y-head_size//2], [x2-head_size, y+head_size//2]], np.int32)
        cv2.fillPoly(img, [pts_right], color)
    else:
        # 開放式箭頭 (<, >)
        cv2.line(img, (x1, y), (x1+head_size, y-head_size//2), color, thickness)
        cv2.line(img, (x1, y), (x1+head_size, y+head_size//2), color, thickness)
        cv2.line(img, (x2, y), (x2-head_size, y-head_size//2), color, thickness)
        cv2.line(img, (x2, y), (x2-head_size, y+head_size//2), color, thickness)

def draw_random_text(img, img_w, img_h):
    """在畫面上隨機灑落文字 (作為負樣本干擾)"""
    num_texts = random.randint(3, 8)
    for _ in range(num_texts):
        text = random.choice(TEXT_VOCAB)
        if random.random() > 0.5:
            text += f" {random.randint(1, 100)}" # 偶爾組合成 "20 ms" 這種格式
        
        # 隨機位置與字體大小
        tx = random.randint(10, img_w - 100)
        ty = random.randint(20, img_h - 20)
        font_scale = random.uniform(0.4, 0.8)
        thickness = random.randint(1, 2)
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)) # 深色系
        
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# ================= 主程式迴圈 =================
for i in range(NUM_IMAGES):
    # 1. 隨機畫布尺寸
    img_w = random.randint(600, 1200)
    img_h = random.randint(400, 800)
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255 # 白底
    
    # 2. 加入隨機背景格線與波形干擾 (簡化版，用線條模擬複雜背景)
    for _ in range(random.randint(2, 5)):
        y_line = random.randint(50, img_h - 50)
        cv2.line(img, (0, y_line), (img_w, y_line), (200, 200, 200), 1) # 淺灰水平線
    for _ in range(random.randint(2, 5)):
        x_line = random.randint(50, img_w - 50)
        cv2.line(img, (x_line, 0), (x_line, img_h), (50, 50, 50), 1) # 深灰垂直線(模擬對齊線)
        
    # 3. 灑落隨機文字 (負樣本)
    draw_random_text(img, img_w, img_h)
    
    # 4. 準備標註資料
    labels = []
    
    # 5. 隨機生成 1 到 3 個雙箭頭
    num_arrows = random.randint(1, 3)
    for _ in range(num_arrows):
        # 隨機屬性
        arrow_w = random.randint(50, img_w // 2) # 箭頭寬度
        x1 = random.randint(20, img_w - arrow_w - 20)
        x2 = x1 + arrow_w
        y = random.randint(30, img_h - 30)
        
        thickness = random.randint(1, 4) # 隨機粗細 (極細到粗)
        head_size = random.randint(10, 25) # 箭頭三角形大小
        
        # 隨機顏色 (90% 機率黑色/深灰，10% 機率紫色/藍色)
        if random.random() > 0.1:
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        else:
            color = (random.randint(150, 200), random.randint(0, 50), random.randint(100, 150)) # BGR的紫色系
            
        # 繪製箭頭
        draw_solid_triangle_arrow(img, x1, x2, y, color, thickness, head_size)
        
        # 🌟 計算 YOLO Bounding Box 🌟
        # 確保框框能完美包住整個箭頭(包含箭頭的上下寬度)
        bbox_x_center = (x1 + x2) / 2.0 / img_w
        bbox_y_center = y / img_h
        bbox_width = (x2 - x1) / img_w
        bbox_height = (head_size * 1.5) / img_h # 框框高度稍微大於箭頭尺寸
        
        # 確保數值在 0~1 之間
        bbox_x_center = max(0.0, min(1.0, bbox_x_center))
        bbox_y_center = max(0.0, min(1.0, bbox_y_center))
        bbox_width = max(0.0, min(1.0, bbox_width))
        bbox_height = max(0.0, min(1.0, bbox_height))
        
        labels.append(f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # 6. 儲存圖片與標註檔
    base_filename = f"wave_v2_{i:04d}"
    cv2.imwrite(f"dataset_v2/images/{base_filename}.jpg", img)
    
    with open(f"dataset_v2/labels/{base_filename}.txt", 'w') as f:
        f.write("\n".join(labels) + "\n")

print(f"✅ 成功生成 {NUM_IMAGES} 張高複雜度波形圖 (包含隨機尺寸、文字干擾、實心/開放箭頭)！")