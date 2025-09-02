#!/bin/bash

set -euo pipefail


#===========激活虚拟环境===========
if [ -f "/home/Maxq/anaconda3/etc/profile.d/conda.sh" ]; then
  source /home/Maxq/anaconda3/etc/profile.d/conda.sh
  conda activate hlcx || {
    echo 'conda虚拟环境激活失败' >&2
    exit 1
  }
else
  echo '未找到conda.sh，conda环境未初始化' >&2
  exit 1
fi

#===========训练模型===========
echo '开始训练模型'

python multivariate_time_series.py  || {
  echo '训练模型程序报错' >&2
  exit 1
}
echo '训练模型结束'
