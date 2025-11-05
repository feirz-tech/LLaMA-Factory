import base64
import requests
import json
import boto3
from io import BytesIO
from tqdm import tqdm
import time
from openai import OpenAI
from prettytable import PrettyTable
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


BUCKET_NAME = "bigdata-kyc-inhouse-vendor"
s3 = boto3.client("s3")
client = OpenAI(base_url=f"http://localhost:23333/v1", api_key="None")
table = PrettyTable()
table.field_names = ["Category", "Accuracy"]

eval_class = ["dark", "glare", "blur", "nondocument", "photocopy", "screenshot", "pass"]
log_path = f"eval_{len(eval_class)}_class_{timestamp}.txt"

PROMPT = "\n<image>You are an ID photo classifier. Your task is to classify an input image into one of the following seven categories: screenshot, photocopy, nondocument, dark, pass, blur, glare.\n\nCategory definitions:\nscreenshot: Images captured directly from a device screen. They often contain elements such as timestamps, taskbars, or app interfaces. These images typically do not show glare, shadows, or any surface irregularities.\nphotocopy: Scanned or copied ID photos. The image is usually flat, may contain large blank or white areas, and could appear in grayscale.\nnondocument: Images that do not contain any ID document at all.\ndark: Images that are too dim or underexposed, making key information (such as faces or ID details) hard to see.\npass: A valid ID photo that does not fall into any of the above problematic categories.\nblur: The loss of image sharpness that makes text, photos, or details appear fuzzy or out of focus, reducing clarity and potentially making critical information unreadable.\nglare: The presence of bright or reflective light on an image that reduces visibility or obscures details, often appearing as white or washed-out areas that lower contrast and make parts of the document harder to read.\n\nClassify the given image into one of these seven categories based on the above definitions.\n\nOutput Format\nAfter completing the classification according to the steps above, you must return the result in the following Dict format (in English):\n{\n    \"category\": \"photocopy\" or \"screenshot\" or \"nondocument\" or \"dark\" or \"pass\" or \"glare\" or \"blur\",\n    \"confidence\": \"a floating-point value between 0 and 1 representing the confidence score of the prediction.\"\n}\n"

# PROMPT = "\n<image>You are an ID photo classifier. Your task is to classify an input image into one of the following five categories: screenshot, photocopy, nondocument, dark, or pass.\n\nCategory definitions:\nscreenshot: Images captured directly from a device screen. They often contain elements such as timestamps, taskbars, or app interfaces. These images typically do not show glare, shadows, or any surface irregularities.\nphotocopy: Scanned or copied ID photos. The image is usually flat, may contain large blank or white areas, and could appear in grayscale.\nnondocument: Images that do not contain any ID document at all.\ndark: Images that are too dim or underexposed, making key information (such as faces or ID details) hard to see.\npass: A valid ID photo that does not fall into any of the above problematic categories.\n\nClassify the given image into one of these five categories based on the above definitions.\n\nOutput Format\nAfter completing the classification according to the steps above, you must return the result in the following Dict format (in English):\n{\n    \"category\": \"photocopy\" or \"screenshot\" or \"nondocument\" or \"dark\" or \"pass\",\n    \"confidence\": \"a floating-point value between 0 and 1 representing the confidence score of the prediction.\"\n}\n"


def s3_to_base64(s3_path):
    """从 s3://bucket/key 下载图片并转成 base64"""
    key = s3_path.replace(f"s3://{BUCKET_NAME}/", "")
    buffer = BytesIO()
    s3.download_fileobj(BUCKET_NAME, key, buffer)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

#eval_class = ["nondocument", "pass", "dark", "photocopy", "screenshot"]
for cls in eval_class:
    TXT_FILE = f"benchmark_{cls}.txt"
    correct = 0

    with open(TXT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

        for idx, image_path in enumerate(lines):
            image_path = lines[idx]
            ext = image_path.split('.')[-1]
            print("download image")
            base64_image = s3_to_base64(image_path)
            print("download successfully!")

            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": PROMPT,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{ext};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
            )
            # https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true
            # f"data:image/{ext};base64,{base64_image}"
            print(f"[{idx+1} / {len(lines)}] {cls}: ", response.choices[0].message.content)
            try:
                ans_dict = json.loads(response.choices[0].message.content)
                if cls == ans_dict["category"]:
                    correct += 1
            except Exception as e:
                print(e)
                if cls in response.choices[0].message.content:
                    correct += 1
        acc = round(correct/len(lines)*100, 2)
        table.add_row([cls, f"{acc}%"])

print(table)

#for s3_path in tqdm(lines, desc="Processing images"):
#    try:
#        img_b64 = s3_to_base64(s3_path)

with open(log_path, "w", encoding="utf-8") as f:
    f.write(str(table))
