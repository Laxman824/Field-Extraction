import cv2
import numpy as np
from utils.text_extraction import extract_value, determine_label
from config.field_patterns import FIELDS

def process_image(image, ocr_model):
    try:
        image = preprocess_image(image)
        img_np = np.array(image)
        result = ocr_model([img_np])

        extracted_fields = {}
        bounding_boxes = []
        height, width = img_np.shape[:2]

        img_all_text = img_np.copy()
        img_fields = img_np.copy()

        # Create bounding boxes for all detected words
        word_id = 0
        for block in result.pages[0].blocks:
            for line in block.lines:
                for word in line.words:
                    x, y, w, h = (word.geometry[0][0], word.geometry[0][1], 
                                word.geometry[1][0] - word.geometry[0][0],
                                word.geometry[1][1] - word.geometry[0][1])
                    box = {
                        "text": word.value,
                        "box": [
                            int(x * width),
                            int(y * height),
                            int((x + w) * width),
                            int((y + h) * height)
                        ],
                        "label": "other",
                        "linking": [],
                        "words": [{
                            "text": word.value,
                            "box": [
                                int(x * width),
                                int(y * height),
                                int((x + w) * width),
                                int((y + h) * height)
                            ]
                        }],
                        "id": word_id
                    }
                    bounding_boxes.append(box)
                    word_id += 1

        # Combine words into lines
        lines = []
        current_line = []
        for word in bounding_boxes:
            if not current_line or abs(word['box'][1] - current_line[-1]['box'][1]) < 20:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        if current_line:
            lines.append(current_line)

        # Extract fields
        for line in lines:
            line_text = ' '.join([word['text'] for word in line])
            field_matches = find_field(line_text, FIELDS)

            for field, start, end in field_matches:
                value_text = line_text[end:].strip()
                next_field_match = find_field(value_text, FIELDS)
                if next_field_match:
                    value_text = value_text[:next_field_match[0][1]].strip()

                value, confidence = extract_value(value_text, field)

                if value and (field not in extracted_fields or confidence > extracted_fields[field][1]):
                    extracted_fields[field] = (value, confidence)

                    field_words = [word for word in line if field.lower() in word['text'].lower()]
                    if field_words:
                        field_words[0]['label'] = determine_label(field)

                    value_words = [word for word in line if word['text'] in value]
                    if value_words:
                        for word in value_words:
                            word['label'] = 'answer'

        # Draw bounding boxes
        for box in bounding_boxes:
            x1, y1, x2, y2 = box['box']
            cv2.rectangle(img_all_text, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_all_text, box['text'], (x1, y1 - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

            if box['label'] in ['question', 'answer']:
                cv2.rectangle(img_fields, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_fields, f"{box['text']} ({box['label']})",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)

        return (img_all_text, img_fields, 
                {k: v[0] for k, v in extracted_fields.items()}, 
                bounding_boxes)

    except Exception as e:
        st.error(f"An error occurred during image processing: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None
