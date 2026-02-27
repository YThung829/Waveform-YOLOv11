from ultralytics import YOLO

# 1. 載入你剛剛訓練出爐的最強權重 (請確認你的 train 資料夾編號，截圖中是 train4)
model_path = "runs/detect/train5/weights/best.pt"
model = YOLO(model_path)

# 2. 指定你要測試的圖片路徑 (請換成你從網路上下載的圖片)
image_path = r"C:\Users\sodar\Projects\WaveForm\test_data\wave_test2.png"

# 3. 讓模型進行預測 (conf=0.5 代表信心水準大於 50% 才畫框)
results = model.predict(source=image_path, conf=0.5)

# 4. 顯示結果！(會自動彈出一個視窗顯示畫好框的圖)
for result in results:
    result.show()
    
    # 如果你想把結果存成圖片，可以把下面這行取消註解
    # result.save("result_output.jpg")

print("✅ 預測完成！")