import cv2
import numpy as np
import random
<<<<<<< HEAD
from pathlib import Path
from scipy.signal import convolve2d

from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.data.data_util import paths_from_lmdb
=======
import torch
from pathlib import Path
from scipy.signal import convolve2d

from basicsr.utils import img2tensor
>>>>>>> 10271d7 (add_psfencoder)
from basicsr.data.transforms import augment


class DIV2KPSFDataset:
    """DIV2K dataset with on-the-fly PSF degradation and XY coordinate channels.

    Each training sample:
      - Loads a DIV2K HR image
      - Randomly crops a patch, records offset
      - Finds the nearest PSF kernel (by Euclidean distance to crop offset)
      - Convolves the crop with the PSF in RGB domain to create LQ
      - Generates XY coordinate channels from global sensor position
      - Applies augmentation (flip/rotate) to GT and LQ+XY together
      - Returns {'lq': (5,H,W), 'gt': (3,H,W), ...}
    """

    def __init__(self, opt):
        self.opt = opt

<<<<<<< HEAD
        # ---- file client (io backend) ----
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.backend_type = self.io_backend_opt['type']
        dataroot_gt = opt['dataroot_gt']

        # ---- Scan DIV2K images ----
        if self.backend_type == 'lmdb':
            # LMDB: read keys from meta_info.txt
            self.io_backend_opt['db_paths'] = [dataroot_gt]
            self.io_backend_opt['client_keys'] = ['gt']
            self.gt_paths = paths_from_lmdb(dataroot_gt)
        else:
            # Disk: scan folder for image files
            exts = {'.png', '.jpg', '.jpeg', '.bmp'}
            self.gt_paths = sorted(
                str(p) for p in Path(dataroot_gt).glob('*')
                if p.suffix.lower() in exts
            )

=======
        # ---- Scan DIV2K images ----
        dataroot_gt = opt['dataroot_gt']
        exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.gt_paths = sorted(
            str(p) for p in Path(dataroot_gt).glob('*')
            if p.suffix.lower() in exts
        )
>>>>>>> 10271d7 (add_psfencoder)
        if len(self.gt_paths) == 0:
            raise FileNotFoundError(f"No image files found in {dataroot_gt}")

        # ---- Load all PSF kernels (only 104 x 31x31x3 ≈ 1.2 MB total) ----
        psf_dir = opt['psf_dir']
        npz_files = sorted(Path(psf_dir).glob('*.npz'))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {psf_dir}")

        self.psf_kernels = []   # list of (31, 31, 3) float32
        centers = []             # list of (cx, cy)

        for f in npz_files:
            data = np.load(str(f))
            self.psf_kernels.append(data['psf'].astype(np.float32))
            centers.append((int(data['cx']), int(data['cy'])))

        self.psf_centers = np.array(centers, dtype=np.float32)  # (N, 2)

        # Fixed sensor dimensions
        self.sensor_w = 4080
        self.sensor_h = 3060

        # ---- Settings ----
        self.gt_size = opt.get('gt_size', 256)
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.use_xy = opt.get('use_xy', True)
        self.phase = opt.get('phase', 'train')
        self.is_train = (self.phase == 'train')

        xy_info = '5ch (RGB+XY)' if self.use_xy else '3ch (RGB)'
        print(f'  [DIV2KPSFDataset] {len(self.gt_paths)} images | {len(self.psf_kernels)} PSF kernels | '
              f'phase={self.phase} | gt_size={self.gt_size} | '
              f'sensor={self.sensor_w}x{self.sensor_h} | lq={xy_info}')

    def __len__(self):
        return len(self.gt_paths)

    def _map_offset_to_sensor(self, offset_x, offset_y, img_w, img_h):
        """Map image pixel offset to sensor coordinate.

        Scales proportionally: image (0..img_w-1) → sensor (0..sensor_w-1).
        """
        sensor_x = offset_x / max(img_w - 1, 1) * (self.sensor_w - 1)
        sensor_y = offset_y / max(img_h - 1, 1) * (self.sensor_h - 1)
        return sensor_x, sensor_y

    def _find_nearest_psf(self, offset_x, offset_y, img_w, img_h):
        """Find index of PSF kernel nearest to the crop offset."""
        sx, sy = self._map_offset_to_sensor(offset_x, offset_y, img_w, img_h)
        dists = np.sum((self.psf_centers - [sx, sy]) ** 2, axis=1)
        return int(np.argmin(dists))

    @staticmethod
    def _psf_convolve_rgb(image, psf):
        """Per-channel PSF convolution in RGB domain."""
<<<<<<< HEAD
=======
        image = image.astype(np.float64)
        psf = psf.astype(np.float64)
>>>>>>> 10271d7 (add_psfencoder)
        convolved = np.zeros_like(image)
        for ch in range(3):
            convolved[:, :, ch] = convolve2d(
                image[:, :, ch], psf[:, :, ch],
                mode='same', boundary='symm'
            )
<<<<<<< HEAD
        return convolved
=======
        return convolved.astype(np.float32)
>>>>>>> 10271d7 (add_psfencoder)

    def _make_xy_grid(self, h, w, sensor_x, sensor_y, step_x, step_y):
        """Create XY coordinate grid using global sensor coordinates.

        Each pixel's (x, y) encodes its absolute position on the sensor,
        normalized to [-1, 1] by sensor dimensions.

        Args:
            h, w: patch height and width
            sensor_x, sensor_y: top-left corner of the patch in sensor coordinates
            step_x, step_y: sensor pixel size of one image pixel (= sensor_w / img_w)

        Returns:
            (H, W, 2) float32 array, x and y both in [-1, 1].
        """
        xs = sensor_x + np.arange(w, dtype=np.float32) * step_x
        ys = sensor_y + np.arange(h, dtype=np.float32) * step_y
        xs_norm = xs / (self.sensor_w - 1) * 2.0 - 1.0
        ys_norm = ys / (self.sensor_h - 1) * 2.0 - 1.0
        xx, yy = np.meshgrid(xs_norm, ys_norm)
        return np.stack([xx, yy], axis=2)

    def _augment_with_xy(self, img_gt, img_lq, xy_grid):
        """Apply flip/rotate augmentation to GT, LQ, and XY together.

        XY channels are corrected after spatial transforms:
        - hflip: negate x
        - vflip: negate y
        - rot90: swap x and y channels
        """
        hflip = self.use_flip and random.random() < 0.5
        vflip = self.use_rot and random.random() < 0.5
        rot90 = self.use_rot and random.random() < 0.5

        imgs = [img_gt, img_lq, xy_grid]
        result = []
        for img in imgs:
            if hflip:
                img = cv2.flip(img, 1)
            if vflip:
                img = cv2.flip(img, 0)
            if rot90:
                img = img.transpose(1, 0, 2)
            result.append(img)

        img_gt, img_lq, xy_grid = result

        # Fix XY channel signs after spatial transform
        if hflip:
            xy_grid[:, :, 0] = -xy_grid[:, :, 0]
        if vflip:
            xy_grid[:, :, 1] = -xy_grid[:, :, 1]
        if rot90:
            xy_grid = xy_grid[:, :, [1, 0]]

        return img_gt, img_lq, xy_grid

    def __getitem__(self, index):
<<<<<<< HEAD
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.gt_paths[index]

        # ---- 1. Load GT (disk or lmdb), BGR → RGB ----
        if self.backend_type == 'lmdb':
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)  # BGR, float32, [0,1]
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)  # → RGB
        else:
            img_gt = cv2.imread(gt_path)
            if img_gt is None:
                raise IOError(f"Failed to read image: {gt_path}")
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
=======
        gt_path = self.gt_paths[index]

        # ---- 1. Load GT, BGR → RGB ----
        img_gt = cv2.imread(gt_path)
        if img_gt is None:
            raise IOError(f"Failed to read image: {gt_path}")
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
>>>>>>> 10271d7 (add_psfencoder)
        h_img, w_img = img_gt.shape[:2]

        # ---- 2. Pad if smaller than gt_size ----
        h_pad = max(0, self.gt_size - h_img)
        w_pad = max(0, self.gt_size - w_img)
        if h_pad > 0 or w_pad > 0:
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
            h_img, w_img = img_gt.shape[:2]

        # ---- 3. Crop + select PSF ----
        if self.is_train:
            top = random.randint(0, h_img - self.gt_size)
            left = random.randint(0, w_img - self.gt_size)
            psf_idx = self._find_nearest_psf(left, top, w_img, h_img)
        else:
            # Validation: deterministic center crop + random PSF
            top = (h_img - self.gt_size) // 2
            left = (w_img - self.gt_size) // 2
            psf_idx = random.randint(0, len(self.psf_kernels) - 1)

        img_gt = img_gt[top:top + self.gt_size, left:left + self.gt_size, :]

        # ---- 4. PSF convolution: GT → LQ (RGB domain) ----
        psf = self.psf_kernels[psf_idx]
        img_lq = self._psf_convolve_rgb(img_gt, psf)
        img_lq = np.clip(img_lq, 0.0, 1.0)

        # ---- 5. Augmentation & XY channel handling ----
        if self.use_xy:
            # Generate XY grid from global sensor coords, then augment together
            sensor_x, sensor_y = self._map_offset_to_sensor(left, top, w_img, h_img)
            step_x = self.sensor_w / w_img
            step_y = self.sensor_h / h_img
            xy_grid = self._make_xy_grid(self.gt_size, self.gt_size,
                                          sensor_x, sensor_y, step_x, step_y)

            if self.is_train and (self.use_flip or self.use_rot):
                img_gt, img_lq, xy_grid = self._augment_with_xy(
                    img_gt, img_lq, xy_grid)

            # Stack LQ + XY → (H, W, 5)
            img_lq = np.concatenate([img_lq, xy_grid], axis=2)
        else:
            # No XY channels: augment GT+LQ only
            if self.is_train and (self.use_flip or self.use_rot):
                img_gt, img_lq = augment(
                    [img_gt, img_lq], self.use_flip, self.use_rot)

        # ---- 6. Convert to tensor (already RGB, no BGR→RGB needed) ----
        img_gt_t = img2tensor(img_gt, bgr2rgb=False, float32=True)   # (3, H, W)
        img_lq_t = img2tensor(img_lq, bgr2rgb=False, float32=True)   # (3|5, H, W)
<<<<<<< HEAD
=======
        kernel_t = torch.from_numpy(psf.transpose(2, 0, 1)).float()  # (3, 31, 31)
>>>>>>> 10271d7 (add_psfencoder)

        return {
            'lq': img_lq_t,
            'gt': img_gt_t,
            'lq_path': gt_path,
            'gt_path': gt_path,
<<<<<<< HEAD
=======
            'psf_cx': int(self.psf_centers[psf_idx][0]),
            'psf_cy': int(self.psf_centers[psf_idx][1]),
            'kernel': kernel_t,
>>>>>>> 10271d7 (add_psfencoder)
        }
