import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from audit_logger.audit_logger import logger_audit_handler as logger
from PIL import Image
import mammoth
from io import BytesIO
import base64
import mammoth
import shutil
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import DocumentStream
from doc_parser.paddleMd import PaddleMd

class ImageWriter(object):
    def __init__(self, document_name, ocr_method = "paddle"):
        if ocr_method not in ('paddle', 'docling'):
            raise Exception(f"Only support Docling or PaddlePaddle  to extract text from image, i,e. docling or paddle")
        
        self.text_handler = self.extract_text_docling
        if ocr_method == "paddle":
            self.paddle = PaddleMd()
            self.text_handler = self.extract_text_paddle

        self._output_dir = os.getenv("TEMP_PATH", None)
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
        self._image_number = 1
        self.image_ocr = []
        self._document_name = document_name

    def extract_text_docling(self, file_path: str) -> str:
        """Extract text from resume file based on extension"""
        converter = DocumentConverter()
        try:
            if isinstance(file_path, bytes):
                buf = BytesIO(file_path)
                source = DocumentStream(name="my_doc", stream=buf)
            else:
                source = file_path
            result = converter.convert(source)

            md = result.document.export_to_markdown()
            logger.info(f"Success markdown -> {file_path[:20]}")
        except Exception as e:
            logger.warning(f"Error proess bank statement to text -> {file_path[:20]}")
            md = ""  
        return md

    def extract_text_paddle(self, file_path: str) -> str:
        """Extract text from resume file based on extension"""
        try:
            result = self.paddle.documentText(file_path, self._document_name, doc_type="image")
            # md = "\n\n".join(list(map(lambda x: f"## Document-{x[0]+1}\n" + x[1]["text"], enumerate(result))))
            md = "\n\n".join(list(map(lambda x: x[1]["text"], enumerate(result))))

            logger.info(f"Success markdown -> {file_path[:20]}, {result[:1]}")
        except Exception as e:
            logger.warning(f"Error proess bank statement to text -> {file_path[:20]}")
            md = ""

        return md
                
    def __call_v1__(self, element):
        extension = element.content_type.partition("/")[2]
        image_filename = f"{self._document_name}{self._image_number}.{extension}"
        if self._output_dir:
            output_file = os.path.join(self._output_dir, image_filename)
        with element.open() as image_source:
            img_raw = image_source.read()
            encoded_src = base64.b64encode(img_raw).decode("ascii")
            if self._output_dir:
                buf = BytesIO(img_raw)
                pil_image = Image.open(buf)#.convert("RGB")
                pil_image.save(output_file)
        
                try:
                    img_txt = self.text_handler(output_file)
                    logger.info(f"Extract text from image file -> {output_file}")
                except:
                    try:
                        img_txt = self.text_handler(img_raw)
                        logger.info(f"Extract text from image bytes -> {img_raw[:10]}")
                    except:
                        img_txt = ""
                        logger.warning("Faile to extract text from image")

                res = {
                    "name": self._document_name,
                    "page": f"page_{self._image_number}",\
                    "text": img_txt,
                    "image": img_raw
                }
                self.image_ocr.append(res)

        self._image_number += 1
        
        return {
            "src": "data:{0};base64,{1}".format(element.content_type, encoded_src)
        }
    
    def __call__(self, element):
        extension = element.content_type.partition("/")[2]
        image_filename = f"{self._document_name}{self._image_number}.{extension}"
        if self._output_dir:
            output_file = os.path.join(self._output_dir, image_filename)
        with element.open() as image_source:
            img_raw = image_source.read()
            encoded_src = base64.b64encode(img_raw).decode("ascii")
            if self._output_dir:
                buf = BytesIO(img_raw)
                pil_image = Image.open(buf)#.convert("RGB")
                pil_image.save(output_file)
        
            try:
                img_txt = self.text_handler(img_raw)
                logger.info(f"Extract text from image bytes -> {img_raw[:10]}")
            except:
                img_txt = ""
                logger.warning("Faile to extract text from image")

            res = {
                "name": self._document_name,
                "page": f"page_{self._image_number}",
                "text": img_txt,
                "image": img_raw
            }
            self.image_ocr.append(res)

        self._image_number += 1
        
        return {
            "src": "data:{0};base64,{1}".format(element.content_type, encoded_src)
        }
        
class DocxMd:

    def __init__(self):
        pass

    def docx_to_text(self, docx_path, name = "doctmp"):
        """
        extract text from word document
        """
        if isinstance(docx_path, str):
            with open(docx_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                text = result.value
                messages = result.messages
                text = text.strip()
        else:
            result = mammoth.extract_raw_text(BytesIO(docx_path))
            text = result.value
            messages = result.messages
            text = text.strip()

        return text

    def docx_to_image(self, docx_path, name = "doctmp"):
        """
        Extract images from word documents
        """

        if isinstance(docx_path, str):
            nm = Path(docx_path).stem
            image_writer = ImageWriter(nm, ocr_method="paddle")
            with open(docx_path, "rb") as docx_file:
                html_rsp = mammoth.convert_to_html(docx_file, convert_image=mammoth.images.img_element(image_writer))
        else:
            nm = name
            image_writer = ImageWriter(nm, ocr_method="paddle")
            html_rsp = mammoth.convert_to_html(BytesIO(docx_path), convert_image=mammoth.images.img_element(image_writer))

        return image_writer.image_ocr

    def parser(self, docx_path, name = "doctmp"):
        img_text = self.docx_to_image(docx_path, name)
        txt = self.docx_to_text(docx_path, name)
        if txt:
            img_text.append({"name": name, "page": "page_text", "text": txt, "image": None})
        
        return img_text
    
    def documentText(self, docx_path, name = "doctmp"):
        return self.parser(docx_path, name)

    async def adocumentText(self, docx_path, name = "doctmp"):
        return self.parser(docx_path, name)
    
if __name__ == "__main__":
    # Example usage
    docx_path = "/home/gs/Downloads/reconcile_data/0525/New Microsoft Word Document.docx"
    # docx_path = "/home/gs/Downloads/reconcile_data/0525/照片 28-05-25 16 42 08.png"
    # docx_path = "./docling_output/New Microsoft Word Document1.png"
    # output_dir = "docling_output"

    inst_docmd = DocxMd()

    with open(docx_path, 'rb') as fo:       
        byteinput = fo.read()
    # d = inst_docmd.documentText(docx_path)
    d = inst_docmd.documentText(byteinput, docx_path)
    # text_content = docx_to_text(docx_path)

    # images = docx_to_image(docx_path, output_dir)
    x = list(map(lambda x: x['page'], d))
    print(f" {d}")
    # import pickle
    # with open("images.pkl", "wb") as fo:
    #     pickle.dump(images, fo)