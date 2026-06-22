"""
合成文档阴影数据集生成器

创建包含文字的文档图像, 添加合成阴影, 用于快速验证实验。
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torchvision.transforms import functional as F


def create_paper_background(width=512, height=512):
    """创建纸张背景"""
    img = np.ones((height, width, 3), dtype=np.float32)

    # 纸张颜色 (浅米色)
    paper_color = np.array([
        random.uniform(0.92, 0.98),
        random.uniform(0.88, 0.96),
        random.uniform(0.82, 0.92)
    ])
    img = img * paper_color.reshape(1, 1, 3)

    # 添加纸张纹理
    noise = np.random.randn(height, width, 3) * 0.015
    img = np.clip(img + noise, 0, 1)

    return img


def add_text_to_document(img_np, text_lines=None):
    """在文档图像上添加文字"""
    img = Image.fromarray((img_np * 255).astype('uint8'))

    if text_lines is None:
        # 生成随机文本
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming science.",
            "Shadow removal improves document readability.",
            "Deep neural networks learn hierarchical features.",
            "Document image processing is an active research area.",
            "Natural language understanding has made progress.",
            "Computer vision enables machines to see and understand.",
            "Transformer architectures achieve state-of-the-art results.",
            "Attention mechanisms allow models to focus on relevant parts.",
            "Data augmentation helps improve model generalization.",
            "ABSTRACT",
            "Introduction",
            "In this paper, we propose a novel approach to document",
            "image shadow removal using text-aware attention mechanisms.",
            "Our method leverages explicit text region detection to",
            "differentially process textual and non-textual areas.",
            "Related Work",
            "Previous methods treat document images as generic natural",
            "images without considering the unique properties of text.",
            "Experimental results demonstrate significant improvements",
            "in both image quality metrics and OCR recognition rates.",
            "Conclusion",
            "We presented a text-aware document shadow removal network",
            "that achieves state-of-the-art performance on benchmarks.",
            "The code and pretrained models will be publicly available.",
            "Method",
            "1. Text Detection Module",
            "2. Shadow Attention Generation",
            "3. Text-Aware Refinement Network",
            "4. Dual-Branch Loss Function",
        ]
        num_lines = random.randint(8, 20)
        text_lines = random.sample(sample_texts, min(num_lines, len(sample_texts)))
        if num_lines > len(sample_texts):
            extras = random.choices(sample_texts, k=num_lines - len(sample_texts))
            text_lines.extend(extras)

    draw = ImageDraw.Draw(img)

    # 尝试加载系统字体, 失败则用默认
    try:
        fonts = [
            ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size=random.randint(12, 18)),
            ImageFont.truetype("C:/Windows/Fonts/times.ttf", size=random.randint(12, 16)),
        ]
    except (IOError, OSError):
        fonts = [ImageFont.load_default()]

    y_offset = random.randint(20, 60)
    for line in text_lines:
        font = random.choice(fonts)
        x_offset = random.randint(30, 80)
        text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))

        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width = len(line) * 8
            text_height = 16

        if y_offset + text_height > 480:
            break

        draw.text((x_offset, y_offset), line, fill=text_color, font=font)
        y_offset += text_height + random.randint(4, 10)

    return np.array(img).astype(np.float32) / 255.0


def add_synthetic_shadow(img_np):
    """添加合成阴影 (不透明度+模糊的暗色区域)"""
    h, w = img_np.shape[:2]

    shadow = np.zeros((h, w), dtype=np.float32)

    # 1-3个阴影区域
    num_shadows = random.randint(1, 3)
    for _ in range(num_shadows):
        # 阴影中心点
        cx = random.randint(-w // 4, w + w // 4)
        cy = random.randint(-h // 4, h + h // 4)

        # 阴影大小
        sx = random.randint(w // 4, w)
        sy = random.randint(h // 6, h // 2)

        # 阴影旋转角度
        angle = random.uniform(-45, 45)

        # 创建椭圆形阴影
        y, x = np.ogrid[:h, :w]

        # 旋转坐标
        theta = np.radians(angle)
        x_rot = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
        y_rot = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)

        ellipse = (x_rot / sx) ** 2 + (y_rot / sy) ** 2
        shadow_region = (ellipse <= 1.0).astype(np.float32)

        # 阴影边缘渐变
        gradient = np.exp(-ellipse / 0.5)
        gradient = np.clip(gradient, 0, 1)

        # 阴影强度
        intensity = random.uniform(0.3, 0.7)
        shadow += gradient * intensity * shadow_region

    # 限制阴影叠加
    shadow = np.clip(shadow, 0, 0.85)

    # 高斯模糊模拟软阴影
    shadow_img = Image.fromarray((shadow * 255).astype('uint8'))
    blur_radius = random.uniform(5, 25)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shadow = np.array(shadow_img).astype(np.float32) / 255.0

    # 应用阴影: output = image * (1 - alpha * shadow)
    shadow_3ch = np.stack([shadow, shadow, shadow], axis=-1)
    shadow_color = np.array([0.2, 0.2, 0.25]).reshape(1, 1, 3)
    shadowed = img_np * (1 - 0.6 * shadow_3ch) + shadow_color * (0.6 * shadow_3ch)

    return np.clip(shadowed, 0, 1), shadow


def generate_dataset(output_dir, num_train=200, num_test=50, img_size=512):
    """生成合成文档阴影数据集"""
    os.makedirs(os.path.join(output_dir, 'train', 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'target'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'target'), exist_ok=True)

    all_ids = []

    for split, num in [('train', num_train), ('test', num_test)]:
        for i in range(num):
            img_id = f"{split}_{i:04d}"

            # 创建文档图像
            clean = create_paper_background(img_size, img_size)
            clean = add_text_to_document(clean)

            # 添加阴影
            shadowed, shadow_mask = add_synthetic_shadow(clean)

            # 保存
            clean_img = Image.fromarray((clean * 255).astype('uint8'))
            shadowed_img = Image.fromarray((shadowed * 255).astype('uint8'))

            clean_img.save(os.path.join(output_dir, split, 'target', f'{img_id}.png'))
            shadowed_img.save(os.path.join(output_dir, split, 'input', f'{img_id}.png'))

            all_ids.append(img_id)

    print(f"Generated {num_train} train + {num_test} test images in {output_dir}")
    return all_ids


if __name__ == '__main__':
    output_dir = './dataset/Synthetic/'
    generate_dataset(output_dir, num_train=200, num_test=50)
    print("Done!")
