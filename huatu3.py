import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PIL import ImagePath

def draw_rounded_rectangle(draw, bbox, radius, fill):
    """绘制圆角矩形"""
    x1, y1, x2, y2 = bbox
    
    # 绘制矩形主体
    draw.rectangle([(x1 + radius, y1), (x2 - radius, y2)], fill=fill)  # 上下
    draw.rectangle([(x1, y1 + radius), (x2, y2 - radius)], fill=fill)  # 左右
    
    # 绘制四个圆角
    draw.ellipse([(x1, y1), (x1 + 2 * radius, y1 + 2 * radius)], fill=fill)  # 左上
    draw.ellipse([(x2 - 2 * radius, y1), (x2, y1 + 2 * radius)], fill=fill)  # 右上
    draw.ellipse([(x1, y2 - 2 * radius), (x1 + 2 * radius, y2)], fill=fill)  # 左下
    draw.ellipse([(x2 - 2 * radius, y2 - 2 * radius), (x2, y2)], fill=fill)  # 右下

def process_semantic_map(image_path, output_path):
    """处理已转换为RGB的语义地图"""
    # RGB值到物体名称的映射
    rgb_to_object = {
        (int(0.94 * 255), int(0.7818 * 255), int(0.66 * 255)): 'chair',
        (int(0.94 * 255), int(0.8868 * 255), int(0.66 * 255)): 'sofa',
        (int(0.8882 * 255), int(0.94 * 255), int(0.66 * 255)): 'potted plant',
        (int(0.7832 * 255), int(0.94 * 255), int(0.66 * 255)): 'bed',
        (int(0.6782 * 255), int(0.94 * 255), int(0.66 * 255)): 'toilet',
        (int(0.66 * 255), int(0.94 * 255), int(0.7468 * 255)): 'tv',
        (int(0.66 * 255), int(0.94 * 255), int(0.8518 * 255)): 'dining-table',
        (int(0.66 * 255), int(0.9232 * 255), int(0.94 * 255)): 'oven',
        (int(0.66 * 255), int(0.8182 * 255), int(0.94 * 255)): 'sink',
        (int(0.66 * 255), int(0.7132 * 255), int(0.94 * 255)): 'refrigerator',
        (int(0.8168 * 255), int(0.66 * 255), int(0.94 * 255)): 'clock',
    }

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像")
    
    # 转换为RGB格式（因为OpenCV使用BGR格式）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    
    # 创建用于绘制的图像副本
    output_img = img_rgb.copy()
    objects = []
    # 为每个颜色创建掩码并处理
    for rgb, object_name in rgb_to_object.items():
        # 创建颜色掩码（允许小的颜色偏差）
        color_tolerance = 5
        lower_bound = np.array([max(0, x - color_tolerance) for x in rgb])
        upper_bound = np.array([min(255, x + color_tolerance) for x in rgb])
        mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        
        # 找到连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        labels = []
        # 处理每个连通区域
        for i in range(1, num_labels):  # 跳过背景（标签0）
            # 获取连通区域的属性
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 50 and object_name!='toilet' and object_name!='potted plant' or (object_name=='potted plant' and area<10):  # 忽略太小的区域
                continue
                
            # 获取中心点
            center_x = int(centroids[i][0])
            center_y = int(centroids[i][1])
            

            labels.append({
                'object':object_name
            })
            objects.append(object_name)
            # 转换为PIL图像以添加文本
            pil_image = Image.fromarray(output_img)
            draw = ImageDraw.Draw(pil_image)
            
            # 设置字体
            try:
                font = ImageFont.truetype("arial.ttf", 11)
            except:
                print("警告：无法加载 arial 字体，使用默认字体")
                font = ImageFont.load_default()
            
            # 获取文本尺寸
            bbox = draw.textbbox((0, 0), object_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 计算文本位置
            x = center_x - text_width // 2
            y = center_y - text_height // 2
            
            # 确保文本位置在图像范围内
            x = max(0, min(x, width - text_width))
            y = max(0, min(y, height - text_height))
            
            # 设置圆角矩形的参数
            padding = 4
            radius = 5  # 圆角半径
            bg_bbox = (
                x - padding,
                y - padding,
                x + text_width + padding,
                y + text_height + padding
            )
            
            # 绘制圆角矩形背景
            draw_rounded_rectangle(draw, bg_bbox, radius, fill=(255,165,0,230))
            
            # 绘制文本
            draw.text((x, y), object_name, fill=(0,0,0), font=font)
            
            # 转回numpy数组
            output_img = np.array(pil_image)
    
    # 保存结果
    cv2.imwrite(output_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    print(objects)
    return output_img,labels,objects

# 使用示例
if __name__ == "__main__":
    input_path = "/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/tmp/dump/collect_val/episodes/thread_0/eps_18/94-sem_map.png"
    output_path = "/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/tmp/dump/collect_val/episodes/thread_0/eps_18/94-labeled_map.png"
    
    try:
        labeled_image,labels,objects = process_semantic_map(input_path, output_path)
        print(labels)
        print(f"标注完成，结果已保存至 {output_path}")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")