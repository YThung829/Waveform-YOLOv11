##########
# 生成單純waveform圖片
##########
import cv2
import numpy as np
import os
import random

# ================= 參數設定 =================
IMG_WIDTH = 800
IMG_HEIGHT = 600
NUM_IMAGES = 1000 # 先生成 10 張來測試看看

# 建立存放資料的資料夾 (YOLO 預設喜歡 images 和 labels 分開)
os.makedirs('dataset/images', exist_ok=True)
os.makedirs('dataset/labels', exist_ok=True)

def draw_waveform(img, y_base, x_points):
    """繪製單條帶有轉態斜率的波形"""
    state = random.choice([-1, 1]) # 隨機決定初始狀態：高(1) 或 低(-1)
    amplitude = 30 # 波形上下起伏的幅度 (總高 60)
    slope_w = 15   # 轉態斜坡的水平寬度 (模擬 Rise/Fall time)
    
    current_x = 0
    current_y = y_base + (state * amplitude)
    
    for x in x_points:
        # 1. 畫水平線到「轉態準備點」
        next_x_start = x - slope_w
        cv2.line(img, (current_x, current_y), (next_x_start, current_y), (0, 0, 0), 2)
        
        # 2. 畫斜線轉態
        state *= -1 
        next_y = y_base + (state * amplitude)
        next_x_end = x + slope_w
        cv2.line(img, (next_x_start, current_y), (next_x_end, next_y), (0, 0, 0), 2)
        
        # 更新當前座標
        current_x = next_x_end
        current_y = next_y
        
    # 3. 畫最後一段水平線到畫布最右側邊緣
    cv2.line(img, (current_x, current_y), (IMG_WIDTH, current_y), (0, 0, 0), 2)

# ================= 主程式迴圈 =================
for i in range(NUM_IMAGES):
    # 1. 建立純白背景畫布
    img = np.ones((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8) * 255
    
    # 2. 隨機決定兩條垂直參考線的 X 座標
    x1 = random.randint(150, 300)
    x2 = random.randint(450, 650)
    
    # 畫出垂直參考線 (貫穿畫布的細線)
    cv2.line(img, (x1, 50), (x1, 550), (0, 0, 0), 1)
    cv2.line(img, (x2, 50), (x2, 550), (0, 0, 0), 1)
    
    # 3. 畫 3 條波形 (Y 軸基準線分別設在 150, 300, 450)
    draw_waveform(img, 150, [x1, x2])
    draw_waveform(img, 300, [x1, random.randint(x1+50, x2-50), x2]) # 中間軌道多一個隨機轉折，增加圖形多樣性
    draw_waveform(img, 450, [x1, x2])
    
    # 4. 畫目標物：雙箭頭
    # 隨機決定箭頭的 Y 軸高度 (避開波形主體)
    y_arrow = random.choice([80, 225, 375, 520]) 
    
    # OpenCV 沒有單一指令畫雙向箭頭，我們從中心點分別往左、往右畫兩個單向箭頭疊加
    center_x = (x1 + x2) // 2
    arrow_size = 15 # 箭頭大小
    tip_len = arrow_size / abs(x2 - center_x) # 換算成 OpenCV 要求的比例參數
    
    cv2.arrowedLine(img, (center_x, y_arrow), (x2, y_arrow), (0, 0, 0), 2, tipLength=tip_len)
    cv2.arrowedLine(img, (center_x, y_arrow), (x1, y_arrow), (0, 0, 0), 2, tipLength=tip_len)
    
    # 5. 🌟 自動計算 YOLO 標註座標 (Normalized) 🌟
    # YOLO 格式: <class_id> <x_center> <y_center> <width> <height>
    # 我們的雙箭頭範圍：X 從 x1 到 x2，Y 約為 y_arrow 上下各 15 像素 (總高 30)
    
    bbox_x_center = (x1 + x2) / 2.0 / IMG_WIDTH
    bbox_y_center = y_arrow / IMG_HEIGHT
    bbox_width = (x2 - x1) / IMG_WIDTH
    bbox_height = 30 / IMG_HEIGHT
    
    # 6. 儲存結果
    # 存圖片
    img_filename = f"dataset/images/wave_{i:03d}.jpg"
    cv2.imwrite(img_filename, img)
    
    # 存標註檔 (類別 0 代表 double_arrow)
    label_filename = f"dataset/labels/wave_{i:03d}.txt"
    with open(label_filename, 'w') as f:
        f.write(f"0 {bbox_x_center:.6f} {bbox_y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

print(f"✅ 成功生成 {NUM_IMAGES} 張波形圖與完美 YOLO 標註檔！請查看 dataset 資料夾。")