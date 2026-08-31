
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

_ROOT = Path(__file__).resolve().parents[2]
_WEIGHTS_DEFAULT = _ROOT / 'third_party/DeepCalib/weights/weights_06_5.61.h5'

CLASSES_FOCAL = list(np.arange(40, 501, 10))
CLASSES_XI    = list(np.arange(0, 61, 1) / 50.)




def build_model(weights_path: Path):
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.layers import Dense, Flatten, Input
    from tensorflow.keras.models import Model

    input_shape = (299, 299, 3)
    main_input  = Input(shape=input_shape, dtype='float32', name='main_input')
    phi_model   = InceptionV3(weights=None, include_top=False,
                              input_tensor=main_input, input_shape=input_shape)
    phi_flattened           = Flatten(name='phi-flattened')(phi_model.output)
    final_output_focal      = Dense(len(CLASSES_FOCAL), activation='softmax',
                                    name='output_focal')(phi_flattened)
    final_output_distortion = Dense(len(CLASSES_XI), activation='softmax',
                                    name='output_distortion')(phi_flattened)
    model = Model(inputs=main_input,
                  outputs=[final_output_focal, final_output_distortion])
    model.load_weights(str(weights_path))
    return model


def predict_params(model, img_bgr: np.ndarray) -> tuple[float, float]:
    from tensorflow.keras.applications.imagenet_utils import preprocess_input

    img = cv2.resize(img_bgr, (299, 299))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float64')
    img = img / 255.
    img = img - 0.5
    img = img * 2.
    img = preprocess_input(img[None])
    pred_f, pred_d = model.predict(img, verbose=0)
    return float(CLASSES_FOCAL[np.argmax(pred_f[0])]),\
           float(CLASSES_XI[np.argmax(pred_d[0])])




def _interp2linear(z, xi, yi, extrapval=np.nan):
    x, y = xi.copy(), yi.copy()
    nrows, ncols, nch = z.shape
    x_bad = (x < 0) | (x > ncols - 1)
    y_bad = (y < 0) | (y > nrows - 1)
    x[x_bad] = 0; y[y_bad] = 0
    ndx = (np.floor(y) * ncols + np.floor(x)).astype('int32')
    d = (x == ncols - 1); x = x - np.floor(x); x[d] += 1; ndx[d] -= 1
    d = (y == nrows - 1); y = y - np.floor(y); y[d] += 1; ndx[d] -= ncols
    t = 1 - y
    channels = []
    for c in range(nch):
        r = z[:, :, c].ravel()
        f = (r[ndx]*t + r[ndx+ncols]*y)*(1-x) + (r[ndx+1]*t + r[ndx+ncols+1]*y)*x
        f[x_bad] = extrapval; f[y_bad] = extrapval
        channels.append(f)
    return np.stack(channels, axis=-1)


def undistort_ucm(img_bgr: np.ndarray, f_px: float, xi: float) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    u0, v0 = W / 2.0, H / 2.0
    gx, gy = np.meshgrid(np.arange(W), np.arange(H))
    Xc = (gx - u0) / f_px
    Yc = (gy - v0) / f_px
    Zc = np.ones_like(Xc)
    a1 = np.sqrt(Zc**2 + Xc**2 + Yc**2)
    a2 = Xc**2 + Yc**2 + Zc**2
    alpha = a1 / a2
    Xs, Ys, Zs = Xc*alpha, Yc*alpha, Zc*alpha
    den = xi * np.sqrt(Xs**2 + Ys**2 + Zs**2) + Zs
    Xd = Xs * f_px / den + u0
    Yd = Ys * f_px / den + v0
    return _interp2linear(img_bgr.astype('float32'), Xd, Yd)


def deepcalib_f_to_pixels(f_dc: float, W: int, H: int) -> float:
    return f_dc * (min(W, H) / 299.0)




def run(
    input_image: str,
    output_dir: str = 'camera_calibration/result',
    weights: str | None = None,
    xi_thresh: float = 0.05,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(input_image)
    if img is None:
        raise ValueError(f'[backend]  {input_image}.')
    H, W = img.shape[:2]

    weights_path = Path(weights) if weights else _WEIGHTS_DEFAULT
    print('[backend]')
    model = build_model(weights_path)

    f_dc, xi = predict_params(model, img)
    f_px = deepcalib_f_to_pixels(f_dc, W, H)
    print(f'[backend]  {f_dc}; {f_px:.1f}; {xi:.4f}.')

    if xi > xi_thresh:
        print('[backend]')
        img_und = np.clip(undistort_ucm(img, f_px, xi), 0, 255).astype('uint8')
    else:
        print(f'[backend]')
        img_und = img

    und_path    = str(out_dir / 'undistorted.jpg')
    params_path = str(out_dir / 'params.json')
    cv2.imwrite(und_path, img_und)

    result = {
        'f_deepcalib': f_dc,
        'f_pixels': f_px,
        'xi': xi,
        'image_W': W,
        'image_H': H,
        'undistorted': und_path,
        'distortion_applied': xi > xi_thresh,
    }
    with open(params_path, 'w') as fp:
        json.dump(result, fp, indent=2)

    result['undistorted_img'] = img_und
    print(f'[backend]  {xi:.4f}.')
    print(f'[backend]  {und_path}.')
    print(f'[backend]  {params_path}.')
    return result


def run_video(
    input_video: str,
    output_dir: str = 'camera_calibration/result',
    weights: str | None = None,
    xi_thresh: float = 0.05,
    n_sample: int = 30,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f'[backend]  {input_video}.')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    sample_count = min(n_sample, total_frames)
    sample_idx   = set(np.linspace(0, total_frames - 1, sample_count, dtype=int))

    weights_path = Path(weights) if weights else _WEIGHTS_DEFAULT
    print('[backend]')
    model = build_model(weights_path)


    f_list, xi_list = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_idx:
            f_dc_i, xi_i = predict_params(model, frame)
            f_list.append(f_dc_i)
            xi_list.append(xi_i)
            print(f'[backend]  {idx:5d}; {f_dc_i}; {xi_i:.4f}.')
        idx += 1

    f_dc = float(np.mean(f_list))
    xi   = float(np.mean(xi_list))
    f_px = deepcalib_f_to_pixels(f_dc, W, H)
    print(f'[backend]  {f_dc:.1f}; {f_px:.1f}; {xi:.4f}.')


    suffix      = Path(input_video).suffix or '.mp4'
    vid_path    = str(out_dir / f'undistorted{suffix}')
    params_path = str(out_dir / 'params.json')
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    writer      = cv2.VideoWriter(vid_path, fourcc, fps, (W, H))

    apply = xi > xi_thresh
    if apply:
        print('[backend]')
    else:
        print('[backend]')

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if apply:
            frame = np.clip(undistort_ucm(frame, f_px, xi), 0, 255).astype('uint8')
        writer.write(frame)
        written += 1
        if written % 100 == 0:
            print(f'[backend]  {written}; {total_frames}.')
    cap.release()
    writer.release()

    result = {
        'f_deepcalib': f_dc,
        'f_pixels': f_px,
        'xi': xi,
        'image_W': W,
        'image_H': H,
        'undistorted_video': vid_path,
        'distortion_applied': apply,
        'total_frames': total_frames,
        'fps': fps,
    }
    with open(params_path, 'w') as fp:
        json.dump(result, fp, indent=2)

    print(f'[backend]  {xi:.4f}.')
    print(f'[backend]  {vid_path}.')
    print(f'[backend]  {params_path}.')
    return result
