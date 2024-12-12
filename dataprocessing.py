import zipfile
import os
import pydicom
from PIL import Image, ImageOps
from tqdm import tqdm
import tempfile
import torch
import clip
import numpy as np
import pandas as pd

# 경로 설정
zip_path = '/home/minhyekj/vinbigdata-chest-xray-abnormalities-detection.zip'
jpg_output_base_path = '/home/minhyekj/chestXray/VinCR/jpg'  # JPG 이미지 저장 기본 폴더 경로
test_output_path = os.path.join(jpg_output_base_path, 'test')
train_output_path = os.path.join(jpg_output_base_path, 'train')
feature_output_path = '/home/minhyekj/chestXray/features'  # 특징 저장 폴더 경로
csv_path = '/home/minhyekj/chestXray/VinCR/train.csv'  # 원본 CSV 파일 경로
output_csv_path = '/home/minhyekj/chestXray/VinCR/transformed_bboxes.csv'  # 변환된 CSV 저장 경로


# JPG 및 특징 저장 폴더 생성
os.makedirs(test_output_path, exist_ok=True)
os.makedirs(train_output_path, exist_ok=True)
os.makedirs(feature_output_path, exist_ok=True)
print(f"JPG 저장 경로 생성 완료: {test_output_path} 및 {train_output_path}")

# 원본 CSV 파일에서 Bounding Box 정보 로드
bbox_df = pd.read_csv(csv_path)

# 변환 후 크기 설정
target_width, target_height = 1024, 1024
original_width, original_height = 2788, 2446

# 변환된 Bounding Box 정보를 저장할 리스트
transformed_data = []

# ZIP 파일에서 DICOM 이미지를 하나씩 가져와 JPG로 변환 및 저장
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    dicom_files = [f for f in zip_ref.namelist() if f.lower().endswith('.dicom')]
    print("ZIP 파일에서 DICOM 파일 목록을 가져왔습니다.")

    for file in tqdm(dicom_files, desc="Converting DICOM to JPG"):
        print(f"Processing file: {file}")
        try:
            # 이미지 ID 추출
            image_id = os.path.splitext(os.path.basename(file))[0]

            # 해당 이미지에 대한 Bounding Box 정보 필터링
            image_bboxes = bbox_df[bbox_df['image_id'] == image_id]

            # 임시 파일에 DICOM 파일 추출
            with zip_ref.open(file) as dicom_file:
                with tempfile.NamedTemporaryFile(suffix=".dicom") as tmp_dicom:
                    tmp_dicom.write(dicom_file.read())
                    tmp_dicom.flush()
                    
                    # 임시 파일에서 DICOM 파일 읽기
                    dicom_data = pydicom.dcmread(tmp_dicom.name)
                    dicom_image = dicom_data.pixel_array

                    # 16비트 이미지를 8비트로 변환
                    dicom_image = (dicom_image / dicom_image.max() * 255).astype(np.uint8)
                    image = Image.fromarray(dicom_image)
                    
                    # 이미지 리사이즈 및 패딩 적용
                    scale = min(target_width / original_width, target_height / original_height)
                    new_width = int(original_width * scale)
                    new_height = int(original_height * scale)
                    pad_x = (target_width - new_width) / 2
                    pad_y = (target_height - new_height) / 2
                    
                    image = ImageOps.pad(image, (target_width, target_height), method=Image.LANCZOS)
                    
                    # 폴더별 JPG 파일 저장 경로 설정
                    if file.startswith('test/'):
                        jpg_filename = image_id + '.jpg'
                        jpg_path = os.path.join(test_output_path, jpg_filename)
                    elif file.startswith('train/'):
                        jpg_filename = image_id + '.jpg'
                        jpg_path = os.path.join(train_output_path, jpg_filename)
                    else:
                        print(f"Unknown folder for file: {file}")
                        continue
                    
                    # JPG 파일로 저장
                    image.save(jpg_path, 'JPEG')
                    print(f"Saved JPG: {jpg_path}")
                    
                    # Bounding Box 좌표 변환 및 저장
                    for _, row in image_bboxes.iterrows():
                        x_min = row['x_min'] * scale + pad_x
                        y_min = row['y_min'] * scale + pad_y
                        x_max = row['x_max'] * scale + pad_x
                        y_max = row['y_max'] * scale + pad_y
                        
                        transformed_data.append({
                            "image_id": image_id,
                            "class_name": row['class_name'],
                            "class_id": row['class_id'],
                            "rad_id": row['rad_id'],
                            "x_min": x_min,
                            "y_min": y_min,
                            "x_max": x_max,
                            "y_max": y_max
                        })
        
        except Exception as e:
            print(f"{file} 변환 중 오류 발생: {e}")

# 변환된 Bounding Box 정보를 새로운 CSV 파일로 저장
transformed_df = pd.DataFrame(transformed_data)
transformed_df.to_csv(output_csv_path, index=False)
print(f"변환된 Bounding Box 좌표를 {output_csv_path}에 저장 완료!")
