# Importing Python dependencies
import os
import uuid
import cv2
from datetime import datetime

# Importing FastAPI dependencies
from fastapi import HTTPException, status
from fastapi.responses import FileResponse

# Importing environment dependencies
import environment.messages as messages
import environment.config as config

#==========================================================
# Face Swap Function
def face_swap_func(image_copy, swapper, model, filename):

    print(messages.FUNCTION_FACE_SWAP)   #DEBUG
    
    #Defining Output File Path
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S") 
    output_file_path = os.path.join(config.STATIC_PATH, f"{filename.split('.')[0]}_{unique_id}_{timestamp}.webp")
    output_file_path = output_file_path.replace(" ", "_")

    
    try:
        # Detect faces in the image
        faces = model.get(image_copy)
        print(messages.FACES_DETECTED, len(faces))   #DEBUG

        if not faces:
            print(messages.NO_FACES_DETECTED)   #DEBUG
            raise ValueError(messages.NO_FACES_DETECTED)

        if len(faces) < 2:
            print(messages.LESS_FACES_DETECTED)   #DEBUG
            raise ValueError(messages.LESS_FACES_DETECTED)
        elif len(faces) > 2:
            print(messages.MORE_FACES_DETECTED)   #DEBUG
            raise ValueError(messages.MORE_FACES_DETECTED)

        face1 = faces[0]
        face2 = faces[1]
        
        # Swap faces - first swap face1 onto face2
        swapped_image = swapper.get(image_copy, face1, face2)        
        # Then swap face2 onto face1's original position
        swapped_image = swapper.get(swapped_image, face2, face1)

        if swapped_image is None or swapped_image.size == 0:
            raise ValueError(messages.FAILED_TO_SWAP_FACES)
        
        # Save swapped image using cv2
        success = cv2.imwrite(output_file_path, swapped_image)
        if not success:
            raise ValueError(f"{messages.FAILED_TO_SAVE_IMAGE} {output_file_path}")
        
        print(f"{messages.IMAGE_SAVED_SUCCESSFULLY} {output_file_path}")   #DEBUG
        image_url = f"{config.IMAGE_URL_PREFIX}{output_file_path}"
        return {
            "message": messages.SUCCESS_SWAPPED_IMAGE,
            "image_url": image_url
        }

    except Exception as e:
        return {
            "message": f"{messages.ERROR_SWAPPING_FACES} {e}",
            "image_url": None
        }

#==========================================================
# Read Image Function
def read_image(request):
    file_name = request.query_params.get("file_name")
    print(messages.FILE_NAME, file_name)   #DEBUG

    if not os.path.exists(file_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=messages.IMAGE_NOT_FOUND
        )
    return FileResponse(
        file_name,
        media_type="image/webp"
        )
