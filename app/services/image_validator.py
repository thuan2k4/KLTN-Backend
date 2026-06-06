from fastapi import HTTPException, UploadFile
# pyrefly: ignore [missing-import]
from PIL import Image
import io
from app.core.config import settings

async def validate_image(file: UploadFile):
    # Kiểm tra content type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid image format",
                "detail": "Only JPG and PNG images are supported",
                "code": "INVALID_IMAGE_FORMAT"
            }
        )

    # Kiểm tra dung lượng
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "File too large",
                "detail": f"Max allowed size is {settings.MAX_UPLOAD_SIZE_MB}MB",
                "code": "FILE_TOO_LARGE"
            }
        )
    
    # Reset file pointer sau khi đọc
    await file.seek(0)

    try:
        img = Image.open(io.BytesIO(content))
        img.verify() # Kiểm tra file có bị hỏng không
        
        # Mở lại để lấy info (sau khi verify thì cần mở lại)
        img = Image.open(io.BytesIO(content))
        width, height = img.size
        
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "width": width,
            "height": height,
            "bytes": content
        }
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Corrupt image",
                "detail": "Could not decode image file",
                "code": "CORRUPT_IMAGE"
            }
        )
