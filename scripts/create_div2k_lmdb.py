"""Create LMDB databases for DIV2K training and validation HR images."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basicsr.utils.lmdb_util import make_lmdb_from_imgs


DATAROOT = Path('./datasets/DIV2K')
SPLITS = ['DIV2K_train_HR', 'DIV2K_valid_HR']
EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def main():
    for split in SPLITS:
        src_dir = DATAROOT / split
        lmdb_dir = DATAROOT / f'{split}.lmdb'

        if not src_dir.is_dir():
            print(f'[SKIP] {src_dir} not found')
            continue

        # Collect image paths and keys
        img_paths = sorted(
            p.name for p in src_dir.glob('*')
            if p.suffix.lower() in EXTS
        )
        keys = sorted(
            p.stem for p in src_dir.glob('*')
            if p.suffix.lower() in EXTS
        )

        if len(img_paths) == 0:
            print(f'[SKIP] No images found in {src_dir}')
            continue

        print(f'\n{"="*60}')
        print(f'Creating {lmdb_dir} ({len(img_paths)} images)...')
        print(f'{"="*60}')

        make_lmdb_from_imgs(
            data_path=str(src_dir),
            lmdb_path=str(lmdb_dir),
            img_path_list=img_paths,
            keys=keys,
            compress_level=1,
            multiprocessing_read=False,
        )

        print(f'[DONE] {lmdb_dir}')

    print('\nAll done!')


if __name__ == '__main__':
    main()
