import requests
from PIL import Image
import io
import base64
import time

def pil_image2base64(pil_img, format="JPEG", base64_only = False):
    """
    Converts a PIL Image to a base64-encoded string.
    
    Args:
        pil_img (Image): PIL Image object
        format (str): Format to save the image (e.g., JPEG, PNG)
    
    Returns:
        str: Base64 encoded string of the image
    """ 
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    if base64_only:
        return img_str
    else:
        return f"data:image/{format.lower()};base64,{img_str}"

def url_image2base64(image_url, format="JPEG", base64_only = False):
    """
    Fetches an image from a URL and returns it as a Base64-encoded string.

    Args:
        image_url (str): URL of the image.
        format (str): Format to convert the image to (e.g., JPEG, PNG).

    Returns:
        str: Base64 encoded string with MIME type.
    """
    # Fetch image from URL
    for _ in range(3):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise error for bad status
            break
        except:
            time.sleep(3)
    # Open image with PIL
    image = Image.open(io.BytesIO(response.content))

    # Convert to desired format if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save image to buffer
    buffered = io.BytesIO()
    image.save(buffered, format=format)

    # Encode to Base64
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if base64_only:
        return img_str
    else:
        return f"data:image/{format.lower()};base64,{img_str}"

def base64_image2pil(base64_str):
    """
    Converts a Base64 encoded image string to a PIL Image object.
    
    Args:
        base64_str (str): Base64 string, optionally with MIME header like:
                          "data:image/png;base64,iVBORw0KG..."

    Returns:
        Image: PIL Image object
    """
    # Remove the MIME type prefix if present
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    # Decode the Base64 string
    image_data = base64.b64decode(base64_str)
    
    # Create an in-memory bytes buffer
    image_buffer = io.BytesIO(image_data)
    
    # Open the image using PIL
    image = Image.open(image_buffer)
    
    return image
    
if __name__ == "__main__":
    img = Image.open("/home/gs/Downloads/image.jpg")
    # pil_image2base64(img)

    url = "https://dam.mediacorp.sg/image/upload/s--HB70_NGL--/c_fill,g_auto,h_468,w_830/f_auto,q_auto/v1/mediacorp/cna/image/2025/07/05/vlcsnap-2025-07-05-12h26m50s050.png"
    b = url_image2base64(url)

    pl = base64_image2pil(b)
    pl.show()