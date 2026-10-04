# Demo

Owner: Team demo + phân tích hình học và cấu trúc cảnh

pip install -e .[demo]
streamlit run demo/app.py

TODO [Team demo + phân tích hình học và cấu trúc cảnh]:

- demo/inference.py: đọc checkpoint Faster R-CNN hoặc YOLO26m, trả box. Không train.
- demo/geometry.py: phân tích hình học và cấu trúc cảnh, mục 3.2 CV-project-tonghop.pdf. Làm ít nhất một hướng: biên, đường thẳng, góc, affine hoặc projective, panorama hoặc căn chỉnh, mặt phẳng hoặc vùng bề mặt chính.
- demo/app.py: hiện box, chồng kết quả hình học, so sánh ảnh vào và ảnh ra. Video thì theo thứ tự thời gian.
