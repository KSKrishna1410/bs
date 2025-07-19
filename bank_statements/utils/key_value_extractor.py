
class KeyValueIdentifierClass:
    def __init__(self, sortedOCRresult, key_info_list,doc_text_lables, tablePosition,documentMasterInfo, docType, y_tolerance=20):
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
        # print("Key info list:", self.key_info_list)
        for text, bbox, confidence in self.sortedOCRresult:
            matched = next((k for k in self.key_info_list if k["key"].lower() == text.lower()), None)
            if matched and matched["key_bounding_box"]:
                self.keys.append((matched["standard_key"], eval(matched["key_bounding_box"]), text))
            else:
                self.values.append((text, bbox))
            # Step 2: Sort by Y first, then X for better alignment detection
        self.keys.sort(key=lambda x: (x[1][0][1], x[1][0][0]))
        # Categorization of Possible Values
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
        
        # Fix: Add proper error handling for documentMasterInfo access
        try:
            if isinstance(self.documentMasterInfo, dict) and key_text in self.documentMasterInfo:
                data_type = self.documentMasterInfo[key_text].get('dataType', 'String')
                dynamicThreshold = 1601 if data_type == 'Double' else 601
            else:
                # Default threshold if documentMasterInfo is not available or key not found
                dynamicThreshold = 601
        except (TypeError, AttributeError, KeyError):
            # Fallback if there's any error accessing documentMasterInfo
            dynamicThreshold = 601
            
        closest_bbox = None
        print(f'Inside Right aligned = {key_text} and dynamicThreshold = {dynamicThreshold}')
        for val_text, val_bbox in self.values:
            normalized_text = cleanedText(val_text)
            if normalized_text in [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', kw.lower().strip()) for kw in self.doc_text_lables]:
                continue
            if val_bbox in self.used_value_bboxes:
                continue  # Skip reused values
            val_x1, val_y1 = val_bbox[0]
            val_x2, val_y2 = val_bbox[1]

            average = statistics.mean([key_x1, key_x2])
            # for y_position_threshold in range(5, 5, 15):
            if key_x2-10 <= val_x1  and key_y2 - 15 <= val_y1 < key_y3 :
            # if key_x1 <= val_x1 <= key_x2 and abs(val_y1 - key_y1) <= self.x_align4_bottom:
            # if val_x1 > average and abs(val_y1 - key_y1) <= self.x_align4_bottom:
                x_distance = val_x1 - key_x2
                if x_distance < min_x_distance:
                    min_x_distance = x_distance
                    closest_value = val_text
                    closest_bbox = val_bbox
                    capturedMethod = 'right_aligned_pair'
                    closest_distance = calculate_distance(key_bbox, closest_bbox)
                    print('Right-aligned match candidate:', key_text, '-->', closest_value, 'with distance', closest_distance, ' -- Dynamic threshold -->', dynamicThreshold)
                    existing_index = next((i for i, kv in enumerate(self.key_value_pairs) if kv["key"] == key_text and kv["method"] == capturedMethod), None)
                    if feilds_pattern.get(key_text):
                        matches = re.findall(feilds_pattern[key_text], closest_value)
                        print(f'Regex match with {closest_value} against pattern {feilds_pattern[key_text]} and matched res is {matches}')
                        if len(matches) == 0:
                            print(f'Skipping this value {closest_value} as Regex pattern Didnt matched')
                            continue
                    if existing_index is not None:
                        print(f'Right-aligned match candidate at index: {existing_index} , {self.key_value_pairs[existing_index]}')
                    for threshold in range(100, dynamicThreshold, 100):
                        print(f'Identified value with in Threshol for { key_text} --> {val_text} and the value is {closest_value} at threshol {threshold} and col_dist {closest_distance}')
                        if closest_value and closest_distance < threshold:
                            print(f'Identified value with in Threshol for { key_text} and the value is {closest_value}')
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
                                print(f"[RIGHT] Matched {key_text} --> {closest_value} within threshold {threshold}: Distance = {closest_distance}")
                                break
                            else:
                                # If it exists, only replace if new distance is smaller
                                existing_entry = self.key_value_pairs[existing_index]
                                existing_distance = existing_entry.get("closest_distance", float('inf'))
                                if closest_distance < existing_distance:
                                    # Replace the existing entry
                                    self.key_value_pairs[existing_index] = {     
                                        "key": key_text,
                                        "value": closest_value,
                                        "key_bbox": key_bbox,
                                        "value_bbox": closest_bbox,
                                        "method": capturedMethod,
                                        "doc_text": currentKey['key'],
                                        "closest_distance": closest_distance
                                    }
        pass

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
        print('Identifying the value for ', key_text)
        for val_text, val_bbox in self.values:
            normalized_text = cleanedText(val_text)
            if normalized_text in [re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', kw.lower().strip()) for kw in self.doc_text_lables]:
                continue
            if val_bbox in self.used_value_bboxes:
                continue  # Skip reused values
            val_x1, val_y1 = val_bbox[0]
            val_y3 = val_bbox[2][1]
            
            # ✅ Filter values directly below key within a tight vertical margin
            y_distance = val_y1 - key_y3
            if not (-5 < y_distance <= self.actual_bottom_threshold):
                continue

            # ✅ Check if it's horizontally aligned with the key
            if not (key_x1 - self.x_align4_bottom <= val_x1 <= key_x2 + self.x_align4_bottom):  # give a little buffer
                continue
            
            # ✅ Check the Value with Regex and pass
            if feilds_pattern.get(key_text):
                matches = re.findall(feilds_pattern[key_text], val_text)
                print(f'Regex match with {val_text} against pattern {feilds_pattern[key_text]} and matched res is {matches}')
                if len(matches) == 0:
                    continue
            # ✅ Pick the closest y-distance only (avoid full Euclidean overshoot)
            if y_distance < min_y_distance:
                min_y_distance = y_distance
                bottom_closest_value = val_text
                closest_bbox = val_bbox
                capturedMethod = 'bottom_aligned_pair'

            if bottom_closest_value == None:
            # ✅ Value is horizontally aligned or slightly below
                if not (key_center_y - 10 <= val_y1 <= key_center_y + 20):  # allow small vertical offset
                    continue

                # ✅ Value starts to the right of the key
                if not (val_x1 >= key_right - 10):  # slight overlap allowed
                    continue

                # ✅ Use minimum horizontal distance instead of Euclidean
                x_distance = val_x1 - key_right
                if x_distance < min_y_distance:
                    min_y_distance = x_distance
                    bottom_closest_value = val_text
                    closest_bbox = val_bbox
                    capturedMethod = 'right_bottom_pair' 
            
            # ✅ Once matched loop is over, add the closest match
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
                print(f"✅ Bottom match for {key_text} --> {bottom_closest_value} with Y-distance: {min_y_distance}")
                self.used_value_bboxes.append(closest_bbox)
                match_found = True
        pass

    def right_bottom_aligned(self):
        # fallback alignment logic here
        pass

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
                        "value": text if key_name == 'Amount_in_words' else match ,
                        "key_bbox": bbox,
                        "value_bbox": bbox,
                        "method": "regex_match",
                        "doc_text": text
                    })

    def getkey_extractedValues(self):
        self.categorize_data()
        for eachKey in self.key_info_list:
            print( 'Getting Key details for these feilds ----> ', eachKey["standard_key"] )
            keybbox = json.loads(eachKey["key_bounding_box"])
            if keybbox is not None and (eachKey["value"] is None or eachKey["standard_key"] == 'Payment Status'):
                if self.docType == 'INVOICE':
                    # print(' printing key details , ', eachKey ) 
                    if (eachKey["standard_key"] == 'Payment Status'):
                        self.right_aligned(eachKey)
                        self.bottom_aligned(eachKey)
                    else:
                        # Fix: Add proper error handling for documentMasterInfo access
                        try:
                            if isinstance(self.documentMasterInfo, dict) and eachKey["standard_key"] in self.documentMasterInfo:
                                data_type = self.documentMasterInfo[eachKey["standard_key"]].get('dataType', 'String')
                                is_double = data_type == "Double"
                            else:
                                # Default to non-double if documentMasterInfo is not available
                                is_double = False
                        except (TypeError, AttributeError, KeyError):
                            # Fallback if there's any error accessing documentMasterInfo
                            is_double = False
                            
                        if not is_double: # Non decimal
                            if self.tablePosition and keybbox[0][1] < self.tablePosition[0][1]: # Search invoice elements above Invoice Line table position item
                                self.right_aligned(eachKey)
                                self.bottom_aligned(eachKey)
                        else:
                            if self.tablePosition and keybbox[0][1]-40 > self.tablePosition[0][1]: # Search Invoice Decimals elements Below table position item
                                self.right_aligned(eachKey)
                                self.bottom_aligned(eachKey)
                else :
                    self.right_aligned(eachKey)
                    # Fix: Add proper error handling for documentMasterInfo access
                    try:
                        if isinstance(self.documentMasterInfo, dict) and eachKey["standard_key"] in self.documentMasterInfo:
                            data_type = self.documentMasterInfo[eachKey["standard_key"]].get('dataType', 'String')
                            is_double = data_type == "Double"
                        else:
                            # Default to non-double if documentMasterInfo is not available
                            is_double = False
                    except (TypeError, AttributeError, KeyError):
                        # Fallback if there's any error accessing documentMasterInfo
                        is_double = False
                        
                    if is_double:
                        self.bottom_aligned(eachKey)
            elif eachKey["value"] is not None:
                print(f"Colon match used for: '{eachKey['standard_key']}'")    
                self.set_key_values(eachKey) #set colon Match values
        self.apply_regex_rules()
        return self.key_value_pairs



def calculate_distance(bbox1, bbox2):
    """Calculate Euclidean distance between two bounding box centers."""
    x1, y1 = (bbox1[0][0] + bbox1[2][0]) / 2, (bbox1[0][1] + bbox1[2][1]) / 2
    x2, y2 = (bbox2[0][0] + bbox2[2][0]) / 2, (bbox2[0][1] + bbox2[2][1]) / 2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) 