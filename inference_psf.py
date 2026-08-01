"""
NAFNet-PSF 推理脚本：输入一张图像，去模糊后输出。

用法：
    python inference_psf.py --input <输入图> --output <输出图> --model <模型权重>

示例：
    # 直接对模糊图像去模糊（需手动生成 XY 坐标）
    python inference_psf.py --input blurry.png --output deblur.png

    # 用干净的图 + PSF 核模拟模糊，再推理（完整 pipeline 测试）
    python inference_psf.py --input clean.png --psf_dir npz_07131 --crop 256 --output deblur.png

    # 整张图推理（不裁剪，自动分块处理）
    python inference_psf.py --input blurry.png --output deblur.png --tile_size 512
"""

import argparse
import cv2
import numpy as np
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal

# ---- 固定参数 ----
IMG_CHANNEL = 5
OUT_CHANNEL = 3
SENSOR_W = 4080
SENSOR_H = 3060


def make_xy_grid(h, w, top, left, img_h, img_w):
    """生成全局传感器坐标系的 XY 网格 (归一化到 [-1, 1])"""
    step_x = SENSOR_W / img_w
    step_y = SENSOR_H / img_h
    # 将图像像素 offset 映射到传感器坐标
    xs = left / max(img_w - 1, 1) * (SENSOR_W - 1) + np.arange(w) * step_x
    ys = top / max(img_h - 1, 1) * (SENSOR_H - 1) + np.arange(h) * step_y
    xs_norm = xs / (SENSOR_W - 1) * 2.0 - 1.0
    ys_norm = ys / (SENSOR_H - 1) * 2.0 - 1.0
    xx, yy = np.meshgrid(xs_norm.astype(np.float32), ys_norm.astype(np.float32))
    return np.stack([xx, yy], axis=2)


def load_model(model_path, width=64, device='cpu'):
    """加载训练好的 NAFNet 模型"""
    # 尝试 NAFNetLocal，失败则用 NAFNet
    for net_cls in [NAFNetLocal, NAFNet]:
        try:
            net = net_cls(
                img_channel=IMG_CHANNEL, out_channel=OUT_CHANNEL,
                width=width,
                enc_blk_nums=[1, 1, 1, 28],
                middle_blk_num=1,
                dec_blk_nums=[1, 1, 1, 1]
            )
            break
        except Exception:
            continue

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    # 处理可能的 'params' key
    if 'params' in state_dict:
        state_dict = state_dict['params']
    # 处理可能的 'params_ema' key
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']

    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    print(f"Model loaded: {model_path}, params: {sum(p.numel() for p in net.parameters()):,}")
    return net


def process_tile(net, tile, top, left, img_h, img_w, device):
    """处理单个 tile：加 XY 通道 → 推理 → 返回去模糊结果"""
    h, w = tile.shape[:2]
    xy = make_xy_grid(h, w, top, left, img_h, img_w)
    lq = np.concatenate([tile.astype(np.float32) / 255.0, xy], axis=2)
    lq_t = torch.from_numpy(lq.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = net(lq_t)  # (1, 3, H, W)

    pred_np = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    pred_np = np.clip(pred_np, 0, 1)
    return (pred_np * 255).astype(np.uint8)


def inference_full(net, img_rgb, device, tile_size=None):
    """全图推理，可选分块"""
    h_img, w_img = img_rgb.shape[:2]

    if tile_size is None or (h_img <= tile_size and w_img <= tile_size):
        return process_tile(net, img_rgb, 0, 0, h_img, w_img, device)

    # 分块推理
    result = np.zeros((h_img, w_img, 3), dtype=np.float32)
    count = np.zeros((h_img, w_img, 1), dtype=np.float32)
    overlap = tile_size // 4

    for y in range(0, h_img, tile_size - overlap):
        for x in range(0, w_img, tile_size - overlap):
            x_end = min(x + tile_size, w_img)
            y_end = min(y + tile_size, h_img)
            x = max(0, x_end - tile_size) if x_end == w_img else x
            y = max(0, y_end - tile_size) if y_end == h_img else y

            tile = img_rgb[y:y + tile_size, x:x + tile_size]
            pred = process_tile(net, tile, y, x, h_img, w_img, device)
            result[y:y + tile_size, x:x + tile_size] += pred.astype(np.float32)
            count[y:y + tile_size, x:x + tile_size] += 1.0

    result = (result / count).astype(np.uint8)
    return result


def main():
    parser = argparse.ArgumentParser(description='NAFNet-PSF 推理')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default='output.png', help='输出图像路径')
    parser.add_argument('--model', type=str, default=None, help='模型权重路径（.pth）；不指定则自动查找最新 checkpoint')
    parser.add_argument('--width', type=int, default=64, help='模型 width 参数')
    parser.add_argument('--psf_dir', type=str, default=None, help='PSF 核目录（若要对干净图先做模糊）')
    parser.add_argument('--tile_size', type=int, default=None, help='分块大小，None 则整张图一次推理')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # ---- 查找模型权重 ----
    model_path = args.model
    if model_path is None:
        exp_dir = Path('experiments/NAFNet-PSF-DIV2K-width64/models')
        pth_files = sorted(exp_dir.glob('*.pth'))
        if not pth_files:
            print(f"Error: 未找到 .pth 文件在 {exp_dir}")
            print("请用 --model 指定权重路径")
            sys.exit(1)
        model_path = str(pth_files[-1])
        print(f"自动选择模型: {model_path}")

    device = torch.device(args.device)
    net = load_model(model_path, width=args.width, device=device)

    # ---- 读取图像 ----
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: 无法读取图像 {args.input}")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img = img_rgb.shape[:2]
    print(f"输入图像: {w_img}x{h_img}")

    # ---- 可选：对干净图做 PSF 模糊（模拟退化） ----
    if args.psf_dir:
        from scipy.signal import convolve2d
        npz_files = sorted(Path(args.psf_dir).glob('*.npz'))
        if not npz_files:
            print(f"Error: {args.psf_dir} 下无 .npz 文件")
            sys.exit(1)
        # 随机选一个 PSF
        import random
        f = random.choice(npz_files)
        data = np.load(str(f))
        psf = data['psf'].astype(np.float32)  # (31, 31, 3)
        cx, cy = int(data['cx']), int(data['cy'])
        print(f"使用 PSF: {f.name}  (cx={cx}, cy={cy})")

        img_blur = np.zeros_like(img_rgb, dtype=np.float32)
        for ch in range(3):
            img_blur[:, :, ch] = convolve2d(
                img_rgb[:, :, ch].astype(np.float32) / 255.0,
                psf[:, :, ch],
                mode='same', boundary='symm'
            )
        img_rgb = np.clip(img_blur * 255, 0, 255).astype(np.uint8)

        # 保存模糊图
        blur_path = args.output.replace('.png', '_blur.png')
        cv2.imwrite(blur_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"模糊图已保存: {blur_path}")

    # ---- 推理 ----
    print(f"推理中 (device={device}, tile_size={args.tile_size})...")
    result = inference_full(net, img_rgb, device, tile_size=args.tile_size)

    # ---- 保存 ----
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"结果已保存: {args.output}")


if __name__ == '__main__':
    main()
