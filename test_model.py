from ultralytics import YOLO

# 載入指定的模型權重 (此為抗干擾特訓版)
model = YOLO("runs/detect/train5/weights/best.pt")

# 輸入測試圖片路徑
image_path = r"C:\Users\sodar\Projects\WaveForm\test_data\wave_test2.png"

# 執行預測 (信心閥值設為 80%)
results = model.predict(source=image_path, conf=0.8)

# 顯示預測結果並畫框
for result in results:
    result.show()
    # result.save("output_result.jpg") # 若需存檔請取消此行註解