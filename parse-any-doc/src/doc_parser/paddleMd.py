from paddleocr import PaddleOCR
import json
import os
import time
import io
from functools import lru_cache
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
from pdf2image import convert_from_bytes

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class PaddleMd:
    def __init__(self):    
        self.ocr = PaddleOCR(use_angle_cls=True) #, lang='en')

    def preprocess_input_byte_image(self, image_bytes, name = "tmpdoc"):
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

        result = self.ocr.predict(img_np)
        output = self.postprocess(result, name)

        return output

    def process_input_byte_pdf(self, pdf_bytes, name = "tmpdoc"):
        # Convert PDF bytes to list of PIL Images
        images = convert_from_bytes(pdf_bytes, dpi=200)  # Higher DPI = better OCR
        results = []

        for i, image in enumerate(images):
            # Convert PIL image to NumPy array (RGB)
            try:
                img_np = np.array(image)

                # OCR expects BGR? No, PaddleOCR handles RGB fine
                result = self.ocr.predict(img_np)
                results.extend(result)
            except:
                results.append(None)
            logger.info(f"process {i+1} / {len(images)}")
        output = self.postprocess(results, name)

        return  output
           
    def get_text(self, input_doc, name = "tmpdoc"):
        """
        Args: 
        input_document PDF or image file or bytes

        Return:
        list of dict with image id & extract text, e.g.

        [{'page': 'page_0', 'text': 'Bank Statement'},
        {'page': 'page_1', 'text': 'Bank Statement 2'}] 
        """
        if isinstance(input_doc, str):
            if Path(input_doc).suffix.lower() not in ['.jpg', '.jpeg', '.png', '.pdf']:
                logger.warning(f"Only support extension, jpg, jpeg, png, pdf")
                return []
            try:
                with open(input_doc, "rb") as fo:
                    input_document = fo.read()
            except Exception as e:
                logger.warning(f"Error in read image: {e}")
                return []
        else:
            input_document = input_doc

        try:
            output = self.preprocess_input_byte_image(input_document, name)
            return output            
        except Exception as e:
            logger.warning(f"Error in safe OCR processing: {e}")
            return []

    def get_text_pdf(self, input_doc, name = "tmpdoc"):
        """
        Args: 
        input_document PDF or image file or bytes

        Return:
        list of dict with image id & extract text, e.g.

        [{'page': 'page_0', 'text': 'Bank Statement'},
        {'page': 'page_1', 'text': 'Bank Statement 2'}] 
        """
        if isinstance(input_doc, str):
            if Path(input_doc).suffix.lower() not in ['.jpg', '.jpeg', '.png', '.pdf']:
                logger.warning(f"Only support extension, jpg, jpeg, png, pdf")
                return []
            
            try:
                with open(input_doc, "rb") as fo:
                    input_document = fo.read()
            except Exception as e:
                logger.warning(f"Error in read image: {e}")
                return []
        else:
            input_document = input_doc

        try:
            output = self.process_input_byte_pdf(input_document, name)
            return output            
        except Exception as e:
            logger.warning(f"Error in safe OCR processing: {e}")
            return []

    def postprocess(self, result, name = "tmpdoc"):
        """
        Args: 
        input_document PDF or image file or bytes

        Return:
        list of dict with image id & extract text, e.g.

        [{'page': 'page_0', 'text': 'Bank Statement'},
        {'page': 'page_1', 'text': 'Bank Statement 2'}] 
        """
        output = []

        try:
            if not result or not result[0]:
                return output
            
            for i, pcont in enumerate(result):
                # Process one page
                text_items = []
                font_h = []
                try:
                    text_info, coords, scores = pcont["rec_texts"], pcont["rec_boxes"],  pcont["rec_scores"]
                    coords = coords.tolist()
                    
                    for wd, pos, confidence in zip(text_info, coords, scores):
                        # Validate coordinates
                        if not isinstance(pos, list) or len(pos) < 4:
                            logger.info(f"Skipping item {wd}, {pos}: Invalid coordinates")
                            continue
                                            
                        text = wd.strip()
                        
                        # Calculate position for sorting
                        try:
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
                lnstr = []
                doc = []
                pos0 = (0, 0, 0, 0)
                for k, (pos, text, confidence) in enumerate(text_items):
                    text = text.strip()
                    if not text:# or confidence < 0.8:
                        # print(text)
                        continue
                    
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
                
                # pid = pcont["page_index"] if pcont["page_index"] else 0
                output.append({"name": name, "page": f"page_{i+1}", "text": doc_str, "image": None})

            return output
            
        except Exception as e:
            logger.warning(f"Error in safe OCR processing: {e}")
            return output
        
    async def adocumentText(self, input_document, name = "doctmp", doc_type = "image"):
        if doc_type not in ("image", "pdf"):
            raise Exception(f"** doc_type MUST be (image, pdf)")
        
        if doc_type == "image":
            return self.get_text(input_document, name)
        else:
            return self.get_text_pdf(input_document, name)
    
    def documentText(self, input_document, name = "doctmp", doc_type = "image"):
        if doc_type == "image":
            return self.get_text(input_document, name)
        else:
            return self.get_text_pdf(input_document, name)
    
# Example usage
if __name__ == "__main__":
    # Replace with your image path
    input_document="/home/gs/Downloads/reconcile_data/0525/照片 28-05-25 16 42 08.png"
    # input_document="/home/gs/Downloads/reconcile_data/0525/New Microsoft Word Document1.pdf"
    # input_document="/home/gs/Downloads/Bank_Statement.pdf"
    # input_document="docling_output/New Microsoft Word Document1.png"
    # input_document="/home/gs/Downloads/Statement.pdf"

    # input_document = "test_image.jpg"
    
    if os.path.exists(input_document):        
        # Process safely
        inst_ocr = PaddleMd()
        st0 = time.time()

        with open(input_document, "rb") as fo:
            input_bytes = fo.read()
        # markdown_result = inst_ocr.documentText(input_document, name = input_document, doc_type = "image")
        markdown_result = inst_ocr.documentText(input_bytes, name = input_document, doc_type = "image")
        # markdown_result = inst_ocr.documentText(input_bytes, name = input_document, doc_type='pdf')
        # markdown_result = inst_ocr.documentText(input_document, name = input_document, doc_type='pdf')
        print(f"Running time: {time.time() - st0}s")
        print("\n" + "="*50)
        print("MARKDOWN RESULT:")
        print("="*50)
        print(markdown_result)
        
        # Save result
        with open("debug_output.md", "w", encoding="utf-8") as f:
            f.write(json.dumps(markdown_result, indent=2))
        print(f"\nResult saved to: debug_output.md")
        
    else:
        print(f"Image file not found: {input_document}")
        print("Please update the input_document variable with a valid image file.")