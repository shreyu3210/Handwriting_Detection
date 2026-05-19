import os
import cv2
import numpy as np
import re
import json
import time
from paddleocr import PaddleOCR

# ==================================================
# CONFIGURATION
# ==================================================
LABEL_MATCH_MAX_X_DISTANCE = 500
LABEL_MATCH_MAX_Y_DISTANCE = 50
MIN_CONFIDENCE = 0.5
OUTPUT_DIR = "structured_output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize OCR (Global so it stays loaded in memory for FastAPI)
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en',
    enable_mkldnn=False
)

def normalize_label(text):
    text = text.upper().strip()
    text = re.sub(r'[.,:;\'"/\-]', '', text).strip()
    
    if text in ['C', 'C1', '(']:
        return 'cash'
    elif text in ['G', 'G7', '6']:
        return 'gpay'
    elif text in ['S', 'S1', '5']:
        return 'system'
    elif text in ['E']:
        return 'expense'
    elif text in ['A']:
        return 'addition'
    return None

def is_amount(text):
    cleaned = re.sub(r'[,\s.]', '', text)
    return bool(re.fullmatch(r'^[-+]?\d+$', cleaned))

def parse_amount(text):
    cleaned = re.sub(r'[,\s.]', '', text)
    try:
        return int(cleaned)
    except Exception:
        return None



def split_glued_label_amount(text):
    label_pattern_sep = r'(C1|c1|G7|g7|S1|s1|[\(ACGSPEacgspe56])'
    label_pattern_nosep = r'([\(ACGSPEacgspe])'
    
    match = re.match(rf'^\s*{label_pattern_sep}\s*[.,:;\'"/\-]+\s*([-+]?\d[0-9,.\s]*)$', text)
    if not match:
        match = re.match(rf'^\s*{label_pattern_sep}\s+([-+]?\d[0-9,.\s]*)$', text)
    if not match:
        match = re.match(rf'^\s*{label_pattern_nosep}\s*([-+]?\d[0-9,.\s]*)$', text)
        
    if match:
        raw_label = match.group(1)
        raw_amount = match.group(2)
        canonical = normalize_label(raw_label)
        
        cleaned_amount = re.sub(r'[,\s.]', '', raw_amount)
        if canonical and bool(re.fullmatch(r'^[-+]?\d+$', cleaned_amount)):
            return canonical, int(cleaned_amount)
    return None, None

def process_image(image_bytes: bytes, date_val: str = None) -> dict:
    """
    Takes raw image bytes, runs OCR, matches labels to values, 
    saves debug artifacts, and returns structured data dict.
    """
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Image file size exceeds 20 MB limit.")

    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if original_image is None:
        raise ValueError("Invalid image file provided.")

    original_h, original_w = original_image.shape[:2]
    print(f"DEBUG: Original image dimensions: {original_w}x{original_h}")
    
    if original_w > 10000 or original_h > 10000:
        raise ValueError("Image dimensions exceed 10000px limit.")

    estimated_mem_mb = (original_w * original_h * 3) / (1024 * 1024)
    print(f"DEBUG: Image shape: {original_image.shape}, Estimated raw memory usage: {estimated_mem_mb:.2f} MB")

    max_dim = 1600
    ratio = 1.0
    if original_w > max_dim or original_h > max_dim:
        ratio = min(max_dim / original_w, max_dim / original_h)
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)
        resized_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_image = original_image.copy()

    resized_h, resized_w = resized_image.shape[:2]

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    result, encimg = cv2.imencode('.jpg', resized_image, encode_param)
    if result:
        final_image = cv2.imdecode(encimg, 1)
    else:
        final_image = resized_image

    cv2.imwrite(os.path.join(OUTPUT_DIR, "resized_input.jpg"), final_image)

    ocr_start_time = time.time()
    results = ocr.predict(final_image)
    ocr_processing_time = time.time() - ocr_start_time
    print(f"DEBUG: OCR processing time: {ocr_processing_time:.2f} seconds")
    
    debug_image = final_image.copy()
    all_boxes = []

    if results and results[0]:
        for result in results:
            if not result:
                continue
            
            boxes = result['dt_polys']
            texts = result['rec_texts']
            scores = result['rec_scores']
            
            for box, text, score in zip(boxes, texts, scores):
                if score < MIN_CONFIDENCE:
                    continue
                    
                pts = np.array(box, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
                
                all_boxes.append({
                    'pts': pts,
                    'text': text,
                    'score': score,
                    'rect': (x, y, w, h),
                    'cx': x + w/2,
                    'cy': y + h/2,
                    'is_used': False
                })

    labels = []
    amounts = []
    glued_entries = {}

    for b in all_boxes:
        text = b['text']
        
        canonical_label = normalize_label(text)
        glued_label, glued_amount = split_glued_label_amount(text)
        
        if canonical_label:
            b['label'] = canonical_label
            labels.append(b)
        elif glued_label and glued_amount is not None:
            glued_entries[glued_label] = glued_amount
            b['is_used'] = True
            b['glued_label'] = glued_label
            b['glued_amount'] = glued_amount
            cv2.polylines(debug_image, [b['pts']], isClosed=True, color=(128, 0, 128), thickness=2) # PURPLE
        elif is_amount(text):
            amounts.append(b)

    structured_data = {
        "date": date_val,
        "metadata": {
            "original_dimensions": [original_w, original_h],
            "resized_dimensions": [resized_w, resized_h],
            "resize_ratio": round(ratio, 4),
            "ocr_processing_time_sec": round(ocr_processing_time, 2)
        },
        "entries": glued_entries
    }

    matched_amounts = set()

    # Spatial label-to-value pairing
    for label_box in labels:
        best_match = None
        min_score = float('inf')
        
        for amount_box in amounts:
            if id(amount_box) in matched_amounts:
                continue
                
            dx = amount_box['cx'] - label_box['cx']
            dy = abs(amount_box['cy'] - label_box['cy'])
            
            if dy <= LABEL_MATCH_MAX_Y_DISTANCE and 0 < dx <= LABEL_MATCH_MAX_X_DISTANCE:
                # Score combines horizontal distance with a heavy penalty for vertical distance
                score = dx + (dy * 10)
                if score < min_score:
                    min_score = score
                    best_match = amount_box
                    
        if best_match:
            val = parse_amount(best_match['text'])
            if val is not None:
                structured_data["entries"][label_box['label']] = val
                matched_amounts.add(id(best_match))
                
                cv2.polylines(debug_image, [label_box['pts']], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.polylines(debug_image, [best_match['pts']], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.line(debug_image, 
                         (int(label_box['cx']), int(label_box['cy'])), 
                         (int(best_match['cx']), int(best_match['cy'])), 
                         (255, 0, 0), 2)
                
                label_box['is_used'] = True
                best_match['is_used'] = True

    # Mark rejected OCR in RED
    for b in all_boxes:
        if not b['is_used']:
            cv2.polylines(debug_image, [b['pts']], isClosed=True, color=(0, 0, 255), thickness=2)

    # Save visualization
    cv2.imwrite(os.path.join(OUTPUT_DIR, "label_matching_debug.jpg"), debug_image)

    # Save JSON file
    with open(os.path.join(OUTPUT_DIR, "structured_data.json"), "w") as f:
        json.dump(structured_data, f, indent=2)

    debug_all_detections = {
        "all_detections": []
    }

    for b in all_boxes:
        category = "other"
        canonical = None
        
        if 'glued_label' in b:
            category = "glued_entry"
            canonical = b['glued_label']
        elif 'label' in b:
            category = "label"
            canonical = b['label']
        elif is_amount(b['text']):
            category = "amount"
            
        debug_all_detections["all_detections"].append({
            "text": b['text'],
            "score": round(float(b['score']), 4),
            "category": category,
            "canonical_label": canonical,
            "bbox": [int(v) for v in b['rect']],
            "matched": b.get('is_used', False)
        })

    with open(os.path.join(OUTPUT_DIR, "debug_all_detections.json"), "w") as f:
        json.dump(debug_all_detections, f, indent=2)

    return structured_data

if __name__ == "__main__":
    # Local fallback for direct execution
    print("Testing local execution...")
    with open("image.png", "rb") as f:
        print(json.dumps(process_image(f.read()), indent=2))
