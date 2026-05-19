# PaddleOCR GPU Troubleshooting Guide

If PaddleOCR is falling back to the CPU instead of using the GPU, it is almost always due to an environment mismatch (usually having the CPU-only version of the paddle library installed instead of the GPU version).

Follow these steps in your terminal to debug and fix the issue.

## Step 1: Run the Diagnostics Script
First, let's check what your Python environment actually sees. Run the following command in your terminal (make sure your virtual environment `.venv` is activated):

```bash
./.venv/bin/python -c "import paddle; print('Compiled with CUDA:', paddle.device.is_compiled_with_cuda()); print('Available GPUs:', paddle.device.get_device())"
```

* **If it prints `Compiled with CUDA: False`**: You have the CPU-only version of PaddlePaddle installed. Proceed to Step 2.
* **If it prints `Compiled with CUDA: True` but fails to find a GPU**: Your CUDA drivers or cuDNN might not be configured correctly in your system's PATH.

## Step 2: Uninstall the CPU Version
If you confirmed you have the CPU version installed, you need to remove it first to avoid conflicts.

```bash
./.venv/bin/pip uninstall paddlepaddle -y
```

## Step 3: Install the GPU Version
Check your CUDA version by running:
```bash
nvidia-smi
```
Look for the `CUDA Version` in the top right corner of the output (e.g., 11.8, 12.1).

Then, install the appropriate `paddlepaddle-gpu` version. Because your environment requires PaddlePaddle >= 3.0 (which is not available on the default pip registry yet), you must install it from the official Paddle mirror:
```bash
uv pip install "paddlepaddle-gpu>=3.3" -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
```

## Step 4: Explicitly Enable GPU in Code
Although `use_gpu=True` is the default in PaddleOCR, it's good practice to explicitly define it if you are troubleshooting. Update your OCR initialization in your Python files:

```python
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en',
    enable_mkldnn=False,
    use_gpu=True  # Explicitly force GPU usage
)
```

## Step 5: Verify the Fix
After reinstalling, run the diagnostics script from Step 1 again. It should now say `Compiled with CUDA: True` and list your GPU (e.g., `gpu:0`). Run your OCR script, and you should see a significant drop in processing time!
