## 环境要求：

1. **操作系统要求：**
   - Windows 11

2. **Python 版本：**
   - Python 3.10.15。

3. **深度学习和机器学习库：**
   - `tensorflow` 2.18.0
   - `pytorch` 2.5.1
   - `torchvision` 0.20.1
   - `torchaudio` 2.5.1
   - `opt-einsum` 3.4.0
   - `scikit-image` 0.20.0

4. **CUDA 和 NVIDIA 工具（如适用）：**
   - `cuda-cudart` 12.4.127
   - `cuda-libraries` 12.4.1
   - `cuda-nvrtc` 12.4.127
   - `libcublas` 12.4.5.8
   - `libcusolver` 11.6.1.9
   - `libcusparse` 12.3.1.170
   - `cudnn` 9.0
   - `pytorch-cuda` 12.4
   
5. **图像和计算机视觉：**
   - `opencv` 4.10.0
   - `opencv-python` 4.10.0.84
   - `imageio` 2.36.0
   - `imageio-ffmpeg` 0.5.1
   - `pyyaml` 6.0.2
   - `cvzone` 1.6.1
   - `shapely` 2.0.6
   - `labelme` 5.2.1
   - `pywavelets` 1.7.0
   
6. **全部安装：**
   ```bash
   pip install -r requirements.txt
   ```
   
## 运行步骤：
   ```bash
   python track.py
   ```

## 输出格式说明
在给定的JSON格式输出中，包含了两个主要部分：

1. **frame_id**: 
   - 表示当前的帧编号。`frame_id: 70` 表示这是第70帧的数据。这个字段通常用于视频分析或动态监测系统中，用来标识数据在时间序列中的位置。

2. **track_status**:
   - 这是一个包含轨道状态的对象，描述了每个轨道的当前状态。每个轨道（如 `track_1`、`track_2`、`track_3`）的状态有两种可能的值：
     - `"occupied"`: 表示该轨道被占用，可能有物体或人正在该轨道上。
     - `"empty"`: 表示该轨道为空，没有物体或人。
   
   对应于 `track_1`, `track_2` 和 `track_3` 的状态：
   - `track_1: "occupied"`: 轨道1是被占用的。
   - `track_2: "empty"`: 轨道2是空的。
   - `track_3: "empty"`: 轨道3也是空的。

```bash
  {
    "frame_id": 70,
    "track_status": {
      "track_1": "occupied",
      "track_2": "empty",
      "track_3": "empty"
    }
   ```