from paddleocr import PaddleOCR
import json
import os

def debug_ocr_result(image_path):
    """
    Debug script to understand PaddleOCR result structure
    """
    print(f"Debugging OCR result for: {image_path}")
    print("=" * 50)
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        # use_gpu=False,
        # show_log=True
    )
    
    try:
        # Perform OCR
        result = ocr.predict(image_path)#, cls=True)
        
        print(f"OCR Result Type: {type(result)}")
        print(f"OCR Result Length: {len(result) if result else 'None'}")
        print()
        
        if result:
            print(f"First element type: {type(result[0])}")
            print(f"First element length: {len(result[0]) if result[0] else 'None'}")
            print()
            
            if result[0]:
                print("Structure of first few items:")
                for i, item in enumerate(result[0][:3]):  # Show first 3 items
                    print(f"\nItem {i}:")
                    print(f"  Type: {type(item)}")
                    print(f"  Length: {len(item) if hasattr(item, '__len__') else 'N/A'}")
                    print(f"  Content: {item}")
                    
                    if isinstance(item, list) and len(item) >= 2:
                        print(f"  Coordinates type: {type(item[0])}")
                        print(f"  Coordinates: {item[0]}")
                        print(f"  Text info type: {type(item[1])}")
                        print(f"  Text info: {item[1]}")
                        
                        if isinstance(item[1], list) and len(item[1]) >= 2:
                            print(f"    Text: '{item[1][0]}'")
                            print(f"    Confidence: {item[1][1]}")
                
                print(f"\nTotal items found: {len(result[0])}")
                
                # Try to extract text safely
                print("\nExtracting text safely:")
                for i, item in enumerate(result[0]):
                    try:
                        if isinstance(item, list) and len(item) >= 2:
                            if isinstance(item[1], list) and len(item[1]) >= 2:
                                text = item[1][0]
                                confidence = item[1][1]
                                print(f"  {i}: '{text}' (confidence: {confidence:.3f})")
                            else:
                                print(f"  {i}: Invalid text info structure: {item[1]}")
                        else:
                            print(f"  {i}: Invalid item structure: {item}")
                    except Exception as e:
                        print(f"  {i}: Error processing item: {e}")
            else:
                print("No text detected in the image.")
        else:
            print("OCR returned None or empty result.")
            
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        import traceback
        traceback.print_exc()

def safe_ocr_to_markdown(image_path):
    """
    Safe OCR processing with extensive error handling
    """
    print(f"\nProcessing {image_path} safely...")
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en')#, use_gpu=False, show_log=False)
    
    try:
        result = ocr.predict(image_path) #ocr(image_path, cls=True)
        markdown = f"# OCR Results from {os.path.basename(image_path)}\n\n"
        
        if not result or not result[0]:
            markdown += "No text detected.\n"
            return markdown
        
        markdown += "## Extracted Text\n\n"
        
        # Process each item safely
        for i, pcont in enumerate(result):
            text_items = []
            font_h = []
            try:
                print(f"** page_index -> {pcont["page_index"]}")
                print(f"** text_type -> {pcont["text_type"]}")
                text_info, coords, scores = pcont["rec_texts"], pcont["rec_boxes"],  pcont["rec_scores"]
                coords = coords.tolist()
                
                for wd, pos, confidence in zip(text_info, coords, scores):
                    # Validate coordinates
                    if not isinstance(pos, list) or len(pos) < 4:
                        print(f"Skipping item {wd}, {pos}: Invalid coordinates")
                        continue
                                        
                    text = wd.strip()
                    
                    # Calculate position for sorting
                    try:
                        x_pos = pos[0] #coords[0][0]  # Top-left x coordinate
                        y_pos = pos[1] #coords[0][1]  # Top-left y coordinate
                        font_h.append(pos[3] - pos[1])
                        text_items.append((pos, text, confidence)) #Y-upper, Y-bottom
                    except (IndexError, TypeError):
                        # If coordinates are malformed, just append without position
                        text_items.append(((0, 0, 0, 0), text, confidence))
                
            except Exception as e:
                print(f"Error processing item {pcont}: {e}")
                continue
        
            # Sort by position
            text_items.sort(key=lambda x: (x[0][1], x[0][0]))
            ln_h = min(font_h)
            
            # Add to markdown
            dpos = 0
            lnstr = []
            doc = []
            pos0 = (0, 0, 0, 0)
            ypre = 0, 0
            for k, (pos, text, confidence) in enumerate(text_items):
                text = text.strip()
                if not text:
                    continue
                if confidence > 0.8:                        
                    markdown += f"{text}\n\n"
                else:
                    markdown += f"*{text}* (confidence: {confidence:.2f})\n\n"
                
                if  k == 0:
                    lnstr.append((text, pos))
                else:
                    yoverlap = min(pos[3],pos0[3])-max(pos[1],pos0[1])                  
                    if yoverlap >= ln_h:
                        lnstr.append((text, pos))
                    else:
                        doc.append(lnstr)
                        lnstr = [(text, pos)]
                pos0 = pos
            
            if lnstr:
                doc.append(lnstr)
            doc_str = ""
            for v in doc:
                v.sort(key=lambda x: (x[1][0]))
                doc_str += " ".join(map(lambda x: x[0], v)) + "\n"
            markdown += "---\n*Generated using PaddleOCR*\n"
        
        return markdown
        
    except Exception as e:
        print(f"Error in safe OCR processing: {e}")
        import traceback
        traceback.print_exc()
        return f"# Error\n\nFailed to process {image_path}: {e}\n"


def test_imagebyte():
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
    import io
    import cv2

    # Initialize PaddleOCR (only once for better performance)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # You can set lang='ch', 'fr', etc.

    # Example: Load image bytes (this could come from a file, web request, etc.)
    input_image = "/home/gs/Downloads/reconcile_data/0525/照片 28-05-25 16 42 08.png"
    with open(input_image, 'rb') as f:
        image_bytes = f.read()

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert PIL Image to NumPy array (OpenCV format: BGR)
    img_np = np.array(image)
    if img_np.ndim == 2:
        # Grayscale to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        # RGBA to RGB
        img_np = img_np[:, :, :3]

    # Optional: Convert RGB (PIL default) to BGR (OpenCV format)
    # PaddleOCR expects RGB, so this may not be necessary
    # But PaddleOCR is generally robust to RGB input
    # img_np = img_np[:, :, ::-1]  # Convert RGB to BGR if needed (but usually not required)

    # Perform OCR
    result = ocr.predict(img_np)

    # Print results
    for line in result:
        print(line)

test_imagebyte()
# Example usage
if __name__ == "__main__":
    # Replace with your image path
    image_path="/home/gs/Downloads/reconcile_data/0525/照片 28-05-25 16 42 08.png"
    image_path="/home/gs/Downloads/reconcile_data/0525/New Microsoft Word Document.pdf"

    # image_path = "test_image.jpg"
    
    if os.path.exists(image_path):
        # Debug the OCR result structure
        # debug_ocr_result(image_path)
        
        # Process safely
        markdown_result = safe_ocr_to_markdown(image_path)
        print("\n" + "="*50)
        print("MARKDOWN RESULT:")
        print("="*50)
        print(markdown_result)
        
        # Save result
        with open("debug_output.md", "w", encoding="utf-8") as f:
            f.write(markdown_result)
        print(f"\nResult saved to: debug_output.md")
        
    else:
        print(f"Image file not found: {image_path}")
        print("Please update the image_path variable with a valid image file.")