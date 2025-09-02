#!/bin/bash


set -euo pipefail
#===========激活虚拟环境===========
if [ -f  "/home/Maxq/anaconda3/etc/profile.d/conda.sh" ]; then
  source /home/Maxq/anaconda3/etc/profile.d/conda.sh
  conda activate hlcx || {
    echo "conda虚拟环境激活失败" >&2
    exit 1
  }
else
  echo "未找到conda.sh,conda环境为初始化" >&2
  exit 1
fi
#===========模型预测===========
echo "开始预测"
python predict.py || {
  echo "模型预测失败" >&2
  exit 1
}
echo '模型预测结束'