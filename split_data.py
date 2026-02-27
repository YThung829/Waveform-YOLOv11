import os
import random
import shutil

# 設定來源路徑
base_dir = 'dataset_v2'
img_dir = os.path.join(base_dir, 'images')
lbl_dir = os.path.join(base_dir, 'labels')

# 建立標準 YOLO 資料夾結構
for split in ['train', 'val']:
    os.makedirs(os.path.join(base_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, 'labels'), exist_ok=True)

# 抓取所有圖片並隨機打亂
images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
random.shuffle(images)

# 80% / 20% 切分
split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def move_files(file_list, split_name):
    for img_name in file_list:
        lbl_name = img_name.replace('.jpg', '.txt')
        
        # 移動圖片與標籤
        shutil.move(os.path.join(img_dir, img_name), 
                    os.path.join(base_dir, split_name, 'images', img_name))
        shutil.move(os.path.join(lbl_dir, lbl_name), 
                    os.path.join(base_dir, split_name, 'labels', lbl_name))

# 執行搬移
move_files(train_imgs, 'train')
move_files(val_imgs, 'val')

# 刪除原本空掉的資料夾
os.rmdir(img_dir)
os.rmdir(lbl_dir)

print(f"✅ 資料集切分完成！Train: {len(train_imgs)} 張, Val: {len(val_imgs)} 張")