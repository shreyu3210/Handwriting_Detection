import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

# ==================================================
# CONFIG SECTION
# ==================================================
ENABLE_MANUAL_ROI = False
ENABLE_HEADER_REMOVAL = True
ENABLE_CONTOUR_FILTERING = False
ENABLE_TEXT_DENSITY_FILTERING = False
ENABLE_INTERACTIVE_ROI = False

# Strategy 1: Manual ROI Config
ROI_X1 = 50
ROI_Y1 = 150
ROI_X2 = 800
ROI_Y2 = 1200

# Strategy 2: Header Removal Config
header_cutoff_percent = 0.22

# ==================================================

def is_amount(text):
    # remove commas, spaces, and periods
    cleaned = re.sub(r'[,\s.]', '', text)
    # detect valid integers or signed integers using regex
    return bool(re.fullmatch(r'^[-+]?\d+$', cleaned))

def is_date(text):
    # detect date patterns like 10-5-2026, 10/05/2026, 9-4-2025
    return bool(re.fullmatch(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', text.strip()))

# Initialize OCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en',
    enable_mkldnn=False
    use_gpu=True
)

image_path = "image.png"
original_image = cv2.imread(image_path)
if original_image is None:
    print(f"Error: Unable to read {image_path}")
    exit(1)

image = original_image.copy()

x_offset = 0
y_offset = 0

# --------------------------------------------------
# STRATEGY 5 — Interactive ROI Selection
# --------------------------------------------------
if ENABLE_INTERACTIVE_ROI:
    print("Please select ROI and press SPACE or ENTER. Cancel with C.")
    roi_selected = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    if roi_selected[2] > 0 and roi_selected[3] > 0:
        ROI_X1, ROI_Y1, w, h = roi_selected
        ROI_X2, ROI_Y2 = ROI_X1 + w, ROI_Y1 + h
        ENABLE_MANUAL_ROI = True

# --------------------------------------------------
# STRATEGY 1 & 2 — Crop Region
# --------------------------------------------------
if ENABLE_MANUAL_ROI:
    image = image[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    x_offset = ROI_X1
    y_offset = ROI_Y1
    cv2.imwrite("roi_preview.jpg", image)
    print("Applied Manual Crop ROI")
elif ENABLE_HEADER_REMOVAL:
    h, w = image.shape[:2]
    cutoff_y = int(h * header_cutoff_percent)
    image = image[cutoff_y:h, 0:w]
    x_offset = 0
    y_offset = cutoff_y
    cv2.imwrite("roi_preview.jpg", image)
    print("Applied Header Removal ROI")

# --------------------------------------------------
# STRATEGY 3 — Handwriting-Focused Contour Detection
# --------------------------------------------------
valid_contour_rects = []
if ENABLE_CONTOUR_FILTERING:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_debug = image.copy()
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h) if h > 0 else 0
            
        # Keep likely handwritten regions
        if area > 100 and 10 < h < 150 and 0.1 < aspect_ratio < 20:
            cv2.rectangle(contour_debug, (x, y), (x+w, y+h), (0, 255, 0), 2) # Accepted
            valid_contour_rects.append((x, y, w, h))
        else:
            cv2.rectangle(contour_debug, (x, y), (x+w, y+h), (0, 0, 255), 2) # Rejected
            
    cv2.imwrite("contours_debug.jpg", contour_debug)
    print("Applied Contour Filtering")

# Run OCR on the processed region
results = ocr.predict(image)

# For visualization
final_ocr_image = original_image.copy()
filtered_regions_image = original_image.copy()

detected_amounts = []
detected_dates = []

all_boxes = []

# Collect all detections and adjust coordinates
for result in results:
    if not result:
        continue
    
    boxes = result['dt_polys']
    texts = result['rec_texts']
    scores = result['rec_scores']
    
    for box, text, score in zip(boxes, texts, scores):
        if score < 0.5:
            continue
            
        pts = np.array(box, dtype=np.int32)
        # Shift back to original image coordinates
        pts[:, 0] += x_offset
        pts[:, 1] += y_offset
        
        x, y, w, h = cv2.boundingRect(pts)
        
        all_boxes.append({
            'pts': pts,
            'text': text,
            'score': score,
            'rect': (x, y, w, h),
            'area': w * h,
            'cx': x + w/2,
            'cy': y + h/2
        })

# --------------------------------------------------
# STRATEGY 4 — Text Density Filtering (Pre-calculation)
# --------------------------------------------------
if ENABLE_TEXT_DENSITY_FILTERING:
    for b in all_boxes:
        nearby = 0
        for other_b in all_boxes:
            if b is other_b: continue
            dist = np.hypot(b['cx'] - other_b['cx'], b['cy'] - other_b['cy'])
            if dist < 120: # 120 pixels radius
                nearby += 1
        b['nearby_count'] = nearby

# Filter and visualize
for b in all_boxes:
    pts, text, score = b['pts'], b['text'], b['score']
    x, y, w, h = b['rect']
    
    rejected = False
    
    # Strategy 3 Filtering Check
    if ENABLE_CONTOUR_FILTERING:
        # Check relative to cropped region
        cx, cy = b['cx'] - x_offset, b['cy'] - y_offset
        inside = any(cx_r <= cx <= cx_r + cw_r and cy_r <= cy <= cy_r + ch_r 
                     for (cx_r, cy_r, cw_r, ch_r) in valid_contour_rects)
        if not inside:
            rejected = True

    # Strategy 4 Filtering Check
    if ENABLE_TEXT_DENSITY_FILTERING and not rejected:
        if b['area'] < 300 and b['nearby_count'] == 0:
            rejected = True

    if rejected:
        cv2.polylines(filtered_regions_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        continue

    # Accepted
    cv2.polylines(filtered_regions_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(final_ocr_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    label = f"{text} ({score:.2f})"
    cv2.putText(final_ocr_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    if is_date(text):
        print(f"DATE: {text}")
        detected_dates.append(text)
    elif is_amount(text):
        print(f"AMOUNT: {text}")
        detected_amounts.append(text)
    else:
        print(f"OTHER: {text}")

cv2.imwrite("filtered_regions.jpg", filtered_regions_image)
cv2.imwrite("final_ocr.jpg", final_ocr_image)
print("Saved final_ocr.jpg and filtered_regions.jpg")