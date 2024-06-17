#！bin/bash
if [ ! -d "toolutils/venv" ]; then
    python3 -m virtualenv toolutils/venv
    source toolutils/venv/bin/activate
    toolutils/venv/bin/pip3 install --upgrade pip
    toolutils/venv/bin/pip3 install -r toolutils/requirements.txt -i http://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
else
    source toolutils/venv/bin/activate
    toolutils/venv/bin/pip3 install -r toolutils/requirements.txt -i http://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn
fi


if [ "$3" == "keras2pb" ];then
    python3 toolutils/keras2pb.py --model_path $1 --output_path $2
fi

