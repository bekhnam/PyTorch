from config import config
from imutils import paths
import shutil
import os

def copy_images(rootDir, destiDir):
    imagePaths = list(paths.list_images(rootDir))
    print(f"[INFO] total images found: {len(imagePaths)}...")
    for imagePath in imagePaths:
        filename = imagePath.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split("_")[0])].strip()

        dirPath = os.path.sep.join([destiDir, label])
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        
        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, p)
    
    currentTotal = list(paths.list_images(destiDir))
    print(f"[INFO] total images copied to {destiDir}: " f"{len(currentTotal)}")

print("[INFO] copying images...")
copy_images(os.path.join(config.DATA_PATH, "training"), config.TRAIN)
copy_images(os.path.join(config.DATA_PATH, "validation"), config.VAL)
copy_images(os.path.join(config.DATA_PATH, "evaluation"), config.TEST)
