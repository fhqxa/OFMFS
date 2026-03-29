import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# 导入项目中的模型
from wrn_mixup_model import wrn28_10
from res_mixup_model import resnet18

class ModelWrapper(torch.nn.Module):
    """包装器，确保模型只返回分类输出"""
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        output = self.model(x)
        # 如果返回的是元组，取第二个元素（分类输出）
        if isinstance(output, tuple):
            return output[1]  # 返回分类输出
        return output

def load_custom_model(checkpoint_path, model_type='WideResNet28_10', num_classes=200, dct_status=False):
    """加载训练好的模型"""
    # 先加载checkpoint来检查实际的类别数
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state']
    
    # 自动检测类别数
    detected_num_classes = None
    for key, value in state.items():
        if 'linear' in key and 'weight' in key:
            if 'L.weight_v' in key:  # WideResNet的distLinear
                detected_num_classes = value.shape[0]
                break
            elif 'weight' in key and len(value.shape) == 2:  # 普通Linear层
                detected_num_classes = value.shape[0]
                break
    
    if detected_num_classes is not None and detected_num_classes != num_classes:
        print(f"⚠️  检测到checkpoint中的类别数为 {detected_num_classes}，但指定的是 {num_classes}")
        print(f"🔄 自动调整为 {detected_num_classes} 个类别")
        num_classes = detected_num_classes
    
    # 创建模型
    if model_type == 'WideResNet28_10':
        model = wrn28_10(num_classes=num_classes, dct_status=dct_status)
    elif model_type == 'ResNet18':
        model = resnet18(num_classes=num_classes)
    
    # 处理DataParallel
    state_keys = list(state.keys())
    if 'module' in state_keys[0]:
        from torch.nn import DataParallel
        model = DataParallel(model)
    
    try:
        model.load_state_dict(state)
        print(f"✅ 成功加载模型，类别数: {num_classes}")
    except RuntimeError as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 请检查模型类型和参数是否正确")
        return None
    
    model.eval()
    
    # 使用包装器确保模型只返回分类输出
    wrapped_model = ModelWrapper(model)
    return wrapped_model

def get_available_layers(model, model_type):
    """获取模型中所有可用的层"""
    layers = {}
    
    if model_type == 'WideResNet28_10':
        # WideResNet的层结构
        layers['conv1'] = model.conv1
        layers['block1_layer0_conv1'] = model.block1.layer[0].conv1
        layers['block1_layer0_conv2'] = model.block1.layer[0].conv2
        layers['block1_layer1_conv1'] = model.block1.layer[1].conv1
        layers['block1_layer1_conv2'] = model.block1.layer[1].conv2
        layers['block1_layer2_conv1'] = model.block1.layer[2].conv1
        layers['block1_layer2_conv2'] = model.block1.layer[2].conv2
        layers['block1_layer3_conv1'] = model.block1.layer[3].conv1
        layers['block1_layer3_conv2'] = model.block1.layer[3].conv2
        
        layers['block2_layer0_conv1'] = model.block2.layer[0].conv1
        layers['block2_layer0_conv2'] = model.block2.layer[0].conv2
        layers['block2_layer1_conv1'] = model.block2.layer[1].conv1
        layers['block2_layer1_conv2'] = model.block2.layer[1].conv2
        layers['block2_layer2_conv1'] = model.block2.layer[2].conv1
        layers['block2_layer2_conv2'] = model.block2.layer[2].conv2
        layers['block2_layer3_conv1'] = model.block2.layer[3].conv1
        layers['block2_layer3_conv2'] = model.block2.layer[3].conv2
        
        layers['block3_layer0_conv1'] = model.block3.layer[0].conv1
        layers['block3_layer0_conv2'] = model.block3.layer[0].conv2
        layers['block3_layer1_conv1'] = model.block3.layer[1].conv1
        layers['block3_layer1_conv2'] = model.block3.layer[1].conv2
        layers['block3_layer2_conv1'] = model.block3.layer[2].conv1
        layers['block3_layer2_conv2'] = model.block3.layer[2].conv2
        layers['block3_layer3_conv1'] = model.block3.layer[3].conv1
        layers['block3_layer3_conv2'] = model.block3.layer[3].conv2
        
        # 最后一层（默认）
        layers['block3_last_conv2'] = model.block3.layer[-1].conv2
        
    elif model_type == 'ResNet18':
        # ResNet18的层结构
        layers['conv1'] = model.conv1
        layers['layer1_0_conv1'] = model.layer1[0].conv1
        layers['layer1_0_conv2'] = model.layer1[0].conv2
        layers['layer1_1_conv1'] = model.layer1[1].conv1
        layers['layer1_1_conv2'] = model.layer1[1].conv2
        
        layers['layer2_0_conv1'] = model.layer2[0].conv1
        layers['layer2_0_conv2'] = model.layer2[0].conv2
        layers['layer2_1_conv1'] = model.layer2[1].conv1
        layers['layer2_1_conv2'] = model.layer2[1].conv2
        
        layers['layer3_0_conv1'] = model.layer3[0].conv1
        layers['layer3_0_conv2'] = model.layer3[0].conv2
        layers['layer3_1_conv1'] = model.layer3[1].conv1
        layers['layer3_1_conv2'] = model.layer3[1].conv2
        
        layers['layer4_0_conv1'] = model.layer4[0].conv1
        layers['layer4_0_conv2'] = model.layer4[0].conv2
        layers['layer4_1_conv1'] = model.layer4[1].conv1
        layers['layer4_1_conv2'] = model.layer4[1].conv2
        
        # 最后一层（默认）
        layers['layer4_last_conv2'] = model.layer4[-1].conv2
    
    return layers

def visualize_model(checkpoint_path, model_type='WideResNet28_10', 
                   num_classes=200, dct_status=False, image_path='./img.png',
                   target_layer_name=None):
    """使用自定义模型进行Grad-CAM++可视化"""
    
    # 加载模型
    model = load_custom_model(checkpoint_path, model_type, num_classes, dct_status)
    if model is None:
        print("❌ 模型加载失败，无法继续可视化")
        return
    
    # 选择目标层 - 需要访问原始模型
    original_model = model.model if hasattr(model, 'model') else model
    
    # 获取所有可用层
    available_layers = get_available_layers(original_model, model_type)
    
    if target_layer_name is None:
        # 使用默认层（最后一层）
        if model_type == 'WideResNet28_10':
            target_layer = original_model.block3.layer[-1].conv2
        elif model_type == 'ResNet18':
            target_layer = original_model.layer4[-1].conv2
        else:
            print(f"❌ 不支持的模型类型: {model_type}")
            return
        print(f"🎯 使用默认目标层: {target_layer}")
    else:
        # 使用指定的层
        if target_layer_name in available_layers:
            target_layer = available_layers[target_layer_name]
            print(f"🎯 使用指定目标层: {target_layer_name} -> {target_layer}")
        else:
            print(f"❌ 找不到指定的层: {target_layer_name}")
            print("💡 可用的层:")
            for layer_name in available_layers.keys():
                print(f"   - {layer_name}")
            return
    
    target_layers = [target_layer]
    
    # 图像预处理 - 保持原始尺寸用于可视化
    if dct_status:
        # DCT模型输入尺寸
        input_size = (56, 56)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(input_size),
        ])
    else:
        if num_classes == 64:  # CIFAR
            input_size = (32, 32)
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
            ])
        else:  # CUB或miniImagenet
            input_size = (84, 84)
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
            ])
    
    # 读取图像
    origin_img = cv2.imread(image_path)
    if origin_img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 保存原始图像用于可视化
    original_img = origin_img.copy()
    
    # 预处理用于模型输入
    rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    crop_img = trans(rgb_img)
    net_input = crop_img.unsqueeze(0)
    
    # 准备画布 - 使用原始图像尺寸
    canvas_img = original_img
    
    # Grad-CAM++
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(net_input)
    grayscale_cam = grayscale_cam[0, :]
    
    # 将热力图调整到原始图像尺寸
    original_height, original_width = canvas_img.shape[:2]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (original_width, original_height))
    
    print(f"📏 热力图尺寸: {grayscale_cam.shape} -> 调整后: {grayscale_cam_resized.shape}")
    print(f"📏 原始图像尺寸: {canvas_img.shape}")
    
    # 可视化 - 使用调整后的热力图
    src_img = np.float32(canvas_img) / 255
    visualization_img = show_cam_on_image(src_img, grayscale_cam_resized, use_rgb=False)
    
    # 显示和保存
    cv2.imshow('Custom Model Feature Map', visualization_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    output_filename = f'{model_type}_visualization.png'
    cv2.imwrite(output_filename, visualization_img)
    print(f"结果已保存为: {output_filename}")

if __name__ == "__main__":
    # 使用CUB数据集训练的模型
    checkpoint_path = './checkpoints/CUB/WideResNet28_10_rotation_5way_1shot_aug/best.tar'
    
    # 示例：可视化不同层
    # 可以选择以下层：
    # - 'conv1': 第一个卷积层
    # - 'block1_layer0_conv1': block1第0层的第一个卷积
    # - 'block1_layer0_conv2': block1第0层的第二个卷积
    # - 'block2_layer0_conv1': block2第0层的第一个卷积
    # - 'block2_layer0_conv2': block2第0层的第二个卷积
    # - 'block3_layer0_conv1': block3第0层的第一个卷积
    # - 'block3_layer0_conv2': block3第0层的第二个卷积
    # - 'block3_last_conv2': 最后一层（默认）
    
    # 使用默认层（最后一层）
    #visualize_model(
    #    checkpoint_path=checkpoint_path,
    #    model_type='WideResNet28_10',
    #    num_classes=200,  # CUB数据集是200个类别
    #    dct_status=False,
    #    image_path='./img.png'
    #)
    
    #使用指定层（取消注释来使用）
    visualize_model(
         checkpoint_path=checkpoint_path,
         model_type='WideResNet28_10',
         num_classes=200,
         dct_status=False,
         image_path='/home/d4090/ldq/cub/005.Crested_Auklet/Crested_Auklet_0070_785261.jpg',
         target_layer_name='block3_layer3_conv1'  # 指定层
     )
