# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')

print("PaddleOCR loaded successfully")