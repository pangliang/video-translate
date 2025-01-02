# 安装
* faster-whisper 模型
    * git clone 排除大文件
        > GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/Systran/faster-distil-whisper-large-v3
    * 单独用工具下载 model.bin 放到 git 目录中覆盖
* pytorch 两种方法选其一
  * 下载后安装
    * https://download.pytorch.org/whl/torch/ 目录中下载 torch-2.5.1+cu124-cp312-cp312-win_amd64.whl
    * pip install ./torch-2.5.1+cu124-cp312-cp312-win_amd64.whl
* ollama
  * https://ollama.com/download
