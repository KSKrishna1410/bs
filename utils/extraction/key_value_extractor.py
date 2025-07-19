import statistics
import re
import json
import math

# Field patterns for validation
feilds_pattern = {
    'GSTIN': r'^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$',
    'PAN': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
    'Amount': r'^\d+(\.\d{1,2})?$',
    'Invoice_No': r'^[A-Za-z0-9\-\/]+$',
    'Date': r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$'
}

# Regex rules for additional extraction
regex_check = [
    {"pattern": r"(?i)rupees\s+([\w\s,]+)\s+only", "key": "Amount_in_words"},
    {"pattern": r"(?i)total\s+amount.*?(\d+(?:\.\d{2})?)", "key": "Total_Amount"},
    {"pattern": r"(?i)invoice\s+no.*?([A-Za-z0-9\-\/]+)", "key": "Invoice_No"},
    {"pattern": r"(?i)date.*?(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", "key": "Date"}
]

def cleanedText(text):
    """Clean text by removing special characters and normalizing whitespace"""
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower().strip())

def calculate_distance(bbox1, bbox2):
    """Calculate Euclidean distance between two bounding box centers."""
    x1, y1 = (bbox1[0][0] + bbox1[2][0]) / 2, (bbox1[0][1] + bbox1[2][1]) / 2
    x2, y2 = (bbox2[0][0] + bbox2[2][0]) / 2, (bbox2[0][1] + bbox2[2][1]) / 2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class KeyValueExtractor:
    """Main class for extracting key-value pairs from documents"""
    def __init__(self, ocr_result, key_info_list, doc_text_labels=None, table_position=None, document_master_info=None, doc_type=None, y_tolerance=20):
        self.identifier = KeyValueIdentifierClass(
            ocr_result,
            key_info_list,
            doc_text_labels or [],
            table_position,
            document_master_info,
            doc_type,
            y_tolerance
        )
    
    def extract(self):
        """Extract key-value pairs from the document"""
        return self.identifier.getkey_extractedValues()

class KeyValueIdentifierClass:
    def __init__(self, sortedOCRresult, key_info_list, doc_text_lables, tablePosition, documentMasterInfo, docType, y_tolerance=20):
        self.sortedOCRresult = sortedOCRresult
        self.key_info_list = key_info_list
        self.y_tolerance = y_tolerance
        self.keys = []
        self.values = []
        self.key_value_pairs = []
        self.used_value_bboxes = []
        self.doc_text_lables = doc_text_lables
        self.actual_bottom_threshold = 30  # Define your bottom threshold
        self.y_align4_right = 20
        self.x_align4_bottom = 5
        self.tablePosition = tablePosition
        self.documentMasterInfo = documentMasterInfo
        self.docType = docType
    
    def categorize_data(self):
        for text, bbox, confidence in self.sortedOCRresult:
            matched = next((k for k in self.key_info_list if k["key"].lower() == text.lower()), None)
            if matched and matched["key_bounding_box"]:
                self.keys.append((matched["standard_key"], eval(matched["key_bounding_box"]), text))
            else:
                self.values.append((text, bbox))
        # Sort by Y first, then X for better alignment detection
        self.keys.sort(key=lambda x: (x[1][0][1], x[1][0][0]))
        self.values.sort(key=lambda x: (x[1][0][1], x[1][0][0]))

    def right_aligned(self, currentKey):
        key_bbox = json.loads(currentKey['key_bounding_box'])
        key_x1, key_y1 = key_bbox[0]  # Top-left
        key_x2, key_y2 = key_bbox[1]  # Top-right
        key_x3, key_y3 = key_bbox[2]  # Bottom-right
        key_x4, key_y4 = key_bbox[3]  # Bottom-left
        closest_value = None
        min_x_distance = float('inf')
        key_text = currentKey['standard_key']
        match_candidates = []
        
        try:
            if isinstance(self.documentMasterInfo, dict) and key_text in self.documentMasterInfo:
                data_type = self.documentMasterInfo[key_text].get('dataType', 'String')
                dynamicThreshold = 1601 if data_type == 'Double' else 601
            else:
                dynamicThreshold = 601
        except (TypeError, AttributeError, KeyError):
            dynamicThreshold = 601
            
        closest_bbox = None
        print(f'Inside Right aligned = {key_text} and dynamicThreshold = {dynamicThreshold}')
        
        for val_text, val_bbox in self.values:
            normalized_text = cleanedText(val_text)
            if normalized_text in [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', kw.lower().strip()) for kw in self.doc_text_lables]:
                continue
            if val_bbox in self.used_value_bboxes:
                continue
                
            val_x1, val_y1 = val_bbox[0]
            val_x2, val_y2 = val_bbox[1]

            if key_x2-10 <= val_x1 and key_y2 - 15 <= val_y1 < key_y3:
                x_distance = val_x1 - key_x2
                if x_distance < min_x_distance:
                    min_x_distance = x_distance
                    closest_value = val_text
                    closest_bbox = val_bbox
                    capturedMethod = 'right_aligned_pair'
                    closest_distance = calculate_distance(key_bbox, closest_bbox)
                    
                    existing_index = next((i for i, kv in enumerate(self.key_value_pairs) if kv["key"] == key_text and kv["method"] == capturedMethod), None)
                    
                    if feilds_pattern.get(key_text):
                        matches = re.findall(feilds_pattern[key_text], closest_value)
                        if len(matches) == 0:
                            continue
                            
                    for threshold in range(100, dynamicThreshold, 100):
                        if closest_value and closest_distance < threshold:
                            if existing_index is None:
                                self.key_value_pairs.append({
                                    "key": key_text,
                                    "value": closest_value,
                                    "key_bbox": key_bbox,
                                    "value_bbox": closest_bbox,
                                    "method": capturedMethod,
                                    "doc_text": currentKey['key'],
                                    "closest_distance": closest_distance
                                })
                                self.used_value_bboxes.append(closest_bbox)
                                break
                            else:
                                existing_entry = self.key_value_pairs[existing_index]
                                existing_distance = existing_entry.get("closest_distance", float('inf'))
                                if closest_distance < existing_distance:
                                    self.key_value_pairs[existing_index] = {     
                                        "key": key_text,
                                        "value": closest_value,
                                        "key_bbox": key_bbox,
                                        "value_bbox": closest_bbox,
                                        "method": capturedMethod,
                                        "doc_text": currentKey['key'],
                                        "closest_distance": closest_distance
                                    }

    def bottom_aligned(self, currentKey):
        key_bbox = json.loads(currentKey['key_bounding_box'])
        key_x1, key_y1 = key_bbox[0]  # Top-left
        key_x2, key_y2 = key_bbox[1]  # Top-right
        key_x3, key_y3 = key_bbox[2]  # Bottom-right
        key_center_y = (key_y1 + key_y3) / 2
        key_right = key_x2
        bottom_closest_value = None
        min_y_distance = float('inf')
        key_text = currentKey['standard_key']
        
        for val_text, val_bbox in self.values:
            normalized_text = cleanedText(val_text)
            if normalized_text in [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', kw.lower().strip()) for kw in self.doc_text_lables]:
                continue
            if val_bbox in self.used_value_bboxes:
                continue
                
            val_x1, val_y1 = val_bbox[0]
            val_y3 = val_bbox[2][1]
            
            y_distance = val_y1 - key_y3
            if not (-5 < y_distance <= self.actual_bottom_threshold):
                continue

            if not (key_x1 - self.x_align4_bottom <= val_x1 <= key_x2 + self.x_align4_bottom):
                continue
            
            if feilds_pattern.get(key_text):
                matches = re.findall(feilds_pattern[key_text], val_text)
                if len(matches) == 0:
                    continue
                    
            if y_distance < min_y_distance:
                min_y_distance = y_distance
                bottom_closest_value = val_text
                closest_bbox = val_bbox
                capturedMethod = 'bottom_aligned_pair'

            if bottom_closest_value == None:
                if not (key_center_y - 10 <= val_y1 <= key_center_y + 20):
                    continue

                if not (val_x1 >= key_right - 10):
                    continue

                x_distance = val_x1 - key_right
                if x_distance < min_y_distance:
                    min_y_distance = x_distance
                    bottom_closest_value = val_text
                    closest_bbox = val_bbox
                    capturedMethod = 'right_bottom_pair' 
            
            if bottom_closest_value:
                closest_distance = calculate_distance(key_bbox, closest_bbox)
                self.key_value_pairs.append({
                    "key": key_text,
                    "value": bottom_closest_value,
                    "key_bbox": key_bbox,
                    "value_bbox": closest_bbox,
                    "method": capturedMethod,
                    "doc_text": currentKey['key'],
                    "closest_distance": closest_distance
                })
                self.used_value_bboxes.append(closest_bbox)
                match_found = True

    def set_key_values(self, key2set):
        self.key_value_pairs.append({
            "key": key2set['standard_key'],
            "value": key2set["value"],
            "key_bbox": json.loads(key2set['key_bounding_box']),
            "value_bbox": json.loads(key2set['key_bounding_box']),
            "method": 'colon_identification',
            "doc_text": key2set['key']
        })

    def apply_regex_rules(self):
        for rule in regex_check:
            pattern = rule["pattern"]
            key_name = rule["key"]
            for text, bbox, confidence in self.sortedOCRresult:
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                for match in matches:
                    self.key_value_pairs.append({
                        "key": key_name,
                        "value": text if key_name == 'Amount_in_words' else match,
                        "key_bbox": bbox,
                        "value_bbox": bbox,
                        "method": "regex_match",
                        "doc_text": text
                    })

    def getkey_extractedValues(self):
        self.categorize_data()
        for eachKey in self.key_info_list:
            keybbox = json.loads(eachKey["key_bounding_box"])
            if keybbox is not None and (eachKey["value"] is None or eachKey["standard_key"] == 'Payment Status'):
                if self.docType == 'INVOICE':
                    if (eachKey["standard_key"] == 'Payment Status'):
                        self.right_aligned(eachKey)
                        self.bottom_aligned(eachKey)
                    else:
                        try:
                            if isinstance(self.documentMasterInfo, dict) and eachKey["standard_key"] in self.documentMasterInfo:
                                data_type = self.documentMasterInfo[eachKey["standard_key"]].get('dataType', 'String')
                                is_double = data_type == "Double"
                            else:
                                is_double = False
                        except (TypeError, AttributeError, KeyError):
                            is_double = False
                            
                        if not is_double:
                            if self.tablePosition and keybbox[0][1] < self.tablePosition[0][1]:
                                self.right_aligned(eachKey)
                                self.bottom_aligned(eachKey)
                        else:
                            if self.tablePosition and keybbox[0][1]-40 > self.tablePosition[0][1]:
                                self.right_aligned(eachKey)
                                self.bottom_aligned(eachKey)
                else:
                    self.right_aligned(eachKey)
                    try:
                        if isinstance(self.documentMasterInfo, dict) and eachKey["standard_key"] in self.documentMasterInfo:
                            data_type = self.documentMasterInfo[eachKey["standard_key"]].get('dataType', 'String')
                            is_double = data_type == "Double"
                        else:
                            is_double = False
                    except (TypeError, AttributeError, KeyError):
                        is_double = False
                        
                    if is_double:
                        self.bottom_aligned(eachKey)
            elif eachKey["value"] is not None:
                self.set_key_values(eachKey)
                
        self.apply_regex_rules()
        return self.key_value_pairs 