"""Подавление цветного шума ниже разрешения: цветность (RGB минус яркость) сглаживается гауссом sigma_c,
яркость — sigma_l. Для наземных данных с пикселем мельче PSF: цветной детали меньше PSF самой широкой полосы
в данных нет, поэтому сглаживание цвета на sigma_c < PSF/2,355 реальных деталей не стирает.
Запуск: python scripts/chroma_denoise.py <вход.tif> <выход.png> sigma_c sigma_l"""
import sys, numpy as np, tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter
src, dst, sc, sl = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
a = tifffile.imread(src).astype(np.float32)
mx = 65535.0 if a.max() > 255 else 255.0
a /= mx
w = np.array([0.2126, 0.7152, 0.0722], np.float32)
Y = a @ w
C = a - Y[..., None]
C = np.stack([gaussian_filter(C[..., k], sc) for k in range(3)], -1)
if sl > 0: Y = gaussian_filter(Y, sl)
out = np.clip(C + Y[..., None], 0, 1)
np.save(dst + '.npy', out) if False else None
tifffile.imwrite(dst.replace('.png', '.tif'), (out * 65535 + 0.5).astype(np.uint16), photometric='rgb', compression='zlib')
Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(dst, optimize=True)
