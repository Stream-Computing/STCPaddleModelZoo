#!/bin/bash

SCRIPT=$(realpath $0)
ROOTPATH=$(dirname $SCRIPT)
G_DATASET_ARRAY=()

DATASET_ARRAY=(
    "open_imagenet"                       \
    "open_squad"                          \
    "open_squad_cmcc"                     \
    "open_lcqmc"                          \
    "open_coco2017"                       \
    "open_cifar"                          \
    "widerface"                           \
    "voxceleb"                            \
    "thucnews"                            \
    "test_icdar2015"                      \
    "open_mot17"                          \
    "open_lmdb"                           \
    "open_ijb_c"                          \
    "open_iiit5k"                         \
    "open_drcd"                           \
    "open_criteo_kaggle"                  \
    "open_cmrc2018"                       \
    "market1501"                          \
    "Kinetics400"                         \
    "fairface"                            \
    "Emotional_Analysis_of_Internet_News" \
    "cat"                                 \
    "attention_ocr"                       \
    "tnews"                               \
    "test_icdar2015_rs50"                 \
    "open_lfw_mtcnn_160"                  \
    "open_cail2019"                       \
    "open_BZNSYP"                         \
    "cmcc_tsm"                            \
    "china-people-daily-ner-corpus"       \
    "open_kits19"                         \
    )

function minio_client_download(){
    echo "Downloading ...."

    mkdir -p download

    if [ ! -f "download/mc" ]; then
        wget https://dl.min.io/client/mc/release/linux-amd64/mc -O download/mc
    fi

    chmod +x download/mc

    mc="./download/mc"

    $mc alias set alias_name/ http://bjsw-storage01.streamcomputing.com:9000 svc-ci Rdxt76*tx

    #echo "Extracting ...."

    mkdir -p toolutils/converted_models
}

function download_extract_open_imagenet(){
    # download open imagenet
    if [ ! -f "download/ILSVRC2012_img_val.tar" ]; then
        $mc cp alias_name/solution/dataset/ILSVRC2012_img_val.tar download/ILSVRC2012_img_val.tar
    fi

    # extract open imagenet
    if [ ! -x "datasets/open_imagenet/ILSVRC2012_img_val" ]; then
        mkdir datasets/open_imagenet/ILSVRC2012_img_val
        tar xf download/ILSVRC2012_img_val.tar -C datasets/open_imagenet/ILSVRC2012_img_val
    fi
}

function download_extract_open_squad(){
    # download open_squad
    if [ ! -f "download/open_squad.tar" ]; then
        $mc cp alias_name/solution/dataset/open_squad.tar download/open_squad.tar
    fi

    # extract open_squad
    if [ ! -f "datasets/open_squad/dev-v1.1.json" ]; then
        tar xf download/open_squad.tar -C datasets/open_squad
    fi
}

function download_extract_open_squad_cmcc(){
    # download open_squad
    if [ ! -f "download/open_squad_cmcc.tar" ]; then
        $mc cp alias_name/solution/dataset/open_squad_cmcc.tar download/open_squad_cmcc.tar
    fi

    # extract open_squad
    if [ ! -f "datasets/open_squad/dev-v1.1.json" ]; then
        tar xf download/open_squad_cmcc.tar -C datasets/open_squad
    fi
}

function download_extract_open_lcqmc(){
    # download open_lcqmc
    if [ ! -f "download/lcqmc.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/lcqmc.tar.gz download/lcqmc.tar.gz
    fi

    # extract open_lcqmc
    if [ ! -d "datasets/open_lcqmc/lcqmc" ]; then
        tar -xf download/lcqmc.tar.gz -C datasets/open_lcqmc
    fi
}

function download_extract_open_coco2017(){
    # download open_coco2017
    if [ ! -f "download/open_coco2017.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/open_coco2017.tar.gz download/open_coco2017.tar.gz
    fi

    # extract open_coco2017
    if [ ! -d "datasets/open_coco2017/val2017" ]; then
        tar -zxf download/open_coco2017.tar.gz -C datasets/open_coco2017
    fi
}

function download_extract_open_cifar(){
    # download open_cifar
    if [ ! -f "download/cifar-100-python.tar" ]; then
        $mc cp alias_name/solution/dataset/cifar-100-python.tar download/cifar-100-python.tar
    fi

    # extract open_cifar
    if [ ! -d "datasets/open_cifar/cifar-100-python" ]; then
        tar xf download/cifar-100-python.tar -C datasets/open_cifar
    fi
}

function download_extract_widerface(){
    # widerface
    if [ ! -f "download/WIDER_val.zip" ]; then
        $mc cp alias_name/solution/raw_dataset/WIDER_val.zip download/WIDER_val.zip
    fi

    # widerface
    if [ ! -f "download/widerface_val_anno.tar.gz" ]; then
        $mc cp alias_name/solution/scenario/cv/face_detection/widerface_val_face_detection/widerface_val_anno.tar.gz download/widerface_val_anno.tar.gz
    fi


    if [ ! -d "datasets/widerface/images" ]; then
        unzip  download/WIDER_val.zip -d datasets/widerface > /dev/null
        mv datasets/widerface/WIDER_val/images datasets/widerface/
        rm -rf datasets/widerface/WIDER_val
    fi

    if [ ! -d "datasets/widerface/ground_truth" ]; then
        tar xf download/widerface_val_anno.tar.gz -C datasets/widerface
    fi
}

function download_extract_voxceleb(){
    # VoxCeleb
    if [ ! -f "download/vox1_test_wav.zip" ]; then
        $mc cp alias_name/solution/dataset/vox1_test_wav.zip download/vox1_test_wav.zip
    fi

    # VoxCeleb
    if [ ! -d "datasets/voxceleb/wav" ]; then
        unzip  download/vox1_test_wav.zip -d datasets/voxceleb > /dev/null
    fi
}

function download_extract_thucnews(){
    # thucnews
    if [ ! -f "download/thucnews.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/thucnews.tar.gz download/thucnews.tar.gz
    fi

    # thucnews
    if [ ! -f "datasets/thucnews/thucnews/cnews.test.txt" ]; then
        tar xf download/thucnews.tar.gz -C datasets/thucnews
    fi
}

function download_extract_test_icdar2015(){
    # test_icdar2015
    if [ ! -f "download/test_icdar2015.tar" ]; then
        $mc cp alias_name/solution/dataset/test_icdar2015.tar download/test_icdar2015.tar
    fi

    # test_icdar2015
    if [ ! -d "datasets/test_icdar2015/ch4_test_images" ]; then
        tar -zxf download/test_icdar2015.tar -C datasets/
    fi
}

function download_extract_open_mot17(){
    # open_mot17
    if [ ! -f "download/MOT17.zip" ]; then
        $mc cp alias_name/solution/dataset/MOT17.zip download/MOT17.zip
    fi

    # open_mot17
    if [ ! -d "datasets/open_mot17/MOT17" ]; then
        unzip  download/MOT17.zip -d datasets/open_mot17 > /dev/null
    fi
}

function download_extract_open_lmdb(){
    # open_lmdb
    if [ ! -f "download/lmdb_evaluation.zip" ]; then
        $mc cp alias_name/solution/dataset/lmdb_evaluation.zip download/lmdb_evaluation.zip
    fi

    # open_lmdb
    if [ ! -d "datasets/open_lmdb/evaluation" ]; then
        unzip download/lmdb_evaluation.zip -d datasets/open_lmdb > /dev/null
    fi
}

function download_extract_open_ijb_c(){
    # open_ijb_c
    if [ ! -f "download/IJBC.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/IJBC.tar.gz download/IJBC.tar.gz
    fi

    # open_ijb_c
    if [ ! -d "datasets/open_ijb_c/IJBC" ]; then
        tar -zxf download/IJBC.tar.gz -C datasets/open_ijb_c
    fi
}

function download_extract_open_iiit5k(){
    # open_iiit5k
    if [ ! -f "download/iiit5k.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/iiit5k.tar.gz download/iiit5k.tar.gz
    fi

    # open_iiit5k
    if [ ! -d "datasets/open_iiit5k/iiit5k" ]; then
        tar -zxf download/iiit5k.tar.gz -C datasets/open_iiit5k
    fi
}

function download_extract_open_drcd(){
    # open_drcd
    if [ ! -f "download/open_drcd.tar" ]; then
        $mc cp alias_name/solution/dataset/open_drcd.tar download/open_drcd.tar
    fi

    # open_drcd
    if [ ! -f "datasets/open_drcd/spiece.model" ]; then
        tar xf download/open_drcd.tar -C datasets/open_drcd
    fi
}

function download_extract_open_criteo_kaggle(){
    # open_criteo_kaggle
    if [ ! -f "download/eval.csv" ]; then
        $mc cp alias_name/solution/dataset/eval.csv download/eval.csv
    fi

    if [ ! -f "datasets/open_criteo_kaggle/eval.csv" ]; then
        cp download/eval.csv datasets/open_criteo_kaggle/eval.csv
    fi
}

function download_extract_open_cmrc2018(){
    # open_cmrc_2018
    if [ ! -f "download/open_cmrc2018.tar" ]; then
        $mc cp alias_name/solution/dataset/open_cmrc2018.tar download/open_cmrc2018.tar
    fi

    # open_cmrc_2018
    if [ ! -f "datasets/open_cmrc2018/vocab.txt" ]; then
        tar xf download/open_cmrc2018.tar -C datasets/open_cmrc2018
    fi
}

function download_extract_market1501(){
    # market1501
    if [ ! -f "download/Market-1501-v15.09.15.zip" ]; then
        $mc cp alias_name/solution/raw_dataset/Market-1501-v15.09.15.zip download/Market-1501-v15.09.15.zip
    fi

    if [ ! -d "datasets/market1501/datasets/Market-1501-v15.09.15" ]; then
        unzip download/Market-1501-v15.09.15.zip -d datasets/market1501/datasets > /dev/null
    fi
}

function download_extract_Kinetics400(){
    # Kinetics_val
    if [ ! -f "download/Kinetics_val.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/Kinetics_val.tar.gz download/Kinetics_val.tar.gz
    fi

    if [ ! -d "datasets/Kinetics400/val_256" ]; then
        tar xf download/Kinetics_val.tar.gz -C datasets/Kinetics400
    fi
}

function download_extract_fairface(){
    # fairface
    if [ ! -f "download/fairface.zip" ]; then
        $mc cp alias_name/solution/dataset/fairface.zip download/fairface.zip
    fi

    if [ ! -d "datasets/fairface/detected_faces" ]; then
        unzip download/fairface.zip -d datasets/fairface > /dev/null
    fi
}

function download_extract_Emotional_Analysis_of_Internet_News(){
    # Emotional_Analysis_of_Internet_News
    if [ ! -f "download/Emotional_Analysis_of_Internet_News.tar" ]; then
        $mc cp alias_name/solution/dataset/Emotional_Analysis_of_Internet_News.tar.gz download/Emotional_Analysis_of_Internet_News.tar.gz
    fi

    # Emotional_Analysis_of_Internet_News
    if [ ! -f "datasets/Emotional_Analysis_of_Internet_News/Test_DataSet.csv" ]; then
        tar xf download/Emotional_Analysis_of_Internet_News.tar.gz -C datasets/Emotional_Analysis_of_Internet_News
    fi
}

function download_extract_cat(){
    # cat
    if [ ! -f "download/CAT_DATASET_02.zip" ]; then
        $mc cp alias_name/solution/dataset/CAT_DATASET_02.zip download/CAT_DATASET_02.zip
    fi

    # cat
    if [ ! -d "datasets/cat/data/CAT_03" ]; then
        unzip  download/CAT_DATASET_02.zip -d datasets/cat/data > /dev/null
    fi
}

function download_extract_attention_ocr(){
    # attention_ocr
    if [ ! -f "download/attention_ocr_test.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/attention_ocr_test.tar.gz download/attention_ocr_test.tar.gz
    fi

    # attention_ocr
    if [ ! -d "datasets/attention_ocr/test_imgs" ]; then
        tar -zxf download/attention_ocr_test.tar.gz -C datasets/attention_ocr
    fi
}

function download_extract_tnews(){
    # tnews
    if [ ! -f "download/tnews.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/tnews.tar.gz download/tnews.tar.gz
    fi

    if [ ! -f "datasets/tnews/tnews/toutiao_category_test.txt" ]; then
        tar xf download/tnews.tar.gz -C datasets/tnews
    fi
}

function download_extract_test_icdar2015_rs50(){
    # test_icdar2015
    if [ ! -d "datasets/test_icdar2015/ch4_test_images" ]; then
        tar -zxf download/test_icdar2015.tar -C datasets/
    fi

    # test_icdar2015_rs50
    if [ ! -d "datasets/test_icdar2015_rs50/ch4_test_images" ]; then
        tar -zxf download/test_icdar2015.tar -C datasets/test_icdar2015_rs50/ --strip-components 1
    fi
}

function download_extract_open_lfw_mtcnn_160(){
    # open_lfw_mtcnn_160
    if [ ! -f "download/lfw_mtcnn_160.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/lfw_mtcnn_160.tar.gz download/lfw_mtcnn_160.tar.gz
    fi

    # open_lfw_mtcnn_160
    if [ ! -d "datasets/open_lfw_mtcnn_160/lfw_mtcnnpy_160" ]; then
        tar -zxf download/lfw_mtcnn_160.tar.gz -C datasets/open_lfw_mtcnn_160
    fi
}

function download_extract_open_cail2019(){
    # open_cail2019-scm
    if [ ! -f "download/open_cail2019.tar" ]; then
        $mc cp alias_name/solution/dataset/open_cail2019.tar download/open_cail2019.tar
    fi

    if [ ! -f "datasets/open_cail2019/test.json" ]; then
        tar xf download/open_cail2019.tar -C datasets/open_cail2019 --strip-components 1
    fi
}

function download_extract_open_BZNSYP(){
    # download fastspeech2_mb_melgan aie model
    if [ ! -f "download/fastspeech2_mb_melgan.tar.gz" ]; then
        $mc cp alias_name/solution/benchmark/gpu_onnx_model/open_model/fastspeech_mb_melgan/fastspeech2_mb_melgan.tar.gz download/fastspeech2_mb_melgan.tar.gz
    fi

    # fastspeech2_mb_melgan aie model
    if [ ! -d "engines/STC/mix_tmp/fastspeech2_mb_melgan" ]; then
        if [ ! -d "engines/STC/mix_tmp" ]; then
            mkdir -p engines/STC/mix_tmp
        fi
        tar -zxf download/fastspeech2_mb_melgan.tar.gz -C engines/STC/mix_tmp/
    fi
}

function download_extract_cmcc_tsm(){
    # TSM something something V2
    if [ ! -f "download/tsm_sthv2_test_data.tar" ]; then
        $mc cp alias_name/solution/dataset/tsm_sthv2_test_data.tar download/tsm_sthv2_test_data.tar
    fi

    # tsm sthv2
    if [ ! -d "datasets/cmcc_tsm/test_data" ]; then
        tar -xf download/tsm_sthv2_test_data.tar -C datasets/cmcc_tsm
    fi
}

function download_extract_china-people-daily-ner-corpus(){
    # china-people-daily-ner-corpus
    if [ ! -f "download/china-people-daily-ner-corpus.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/china-people-daily-ner-corpus.tar.gz download/china-people-daily-ner-corpus.tar.gz
    fi

    if [ ! -f "datasets/china-people-daily-ner-corpus/vocab.txt" ]; then
        tar xf download/china-people-daily-ner-corpus.tar.gz -C datasets/
    fi
}

function download_extract_open_kits19(){
    # open_kits19_copy_case185>case400
    if [ ! -f "download/kits19.tar.gz" ]; then
        $mc cp alias_name/solution/dataset/kits19.tar.gz download/kits19.tar.gz
    fi

    # open_kits19_copy_case185>case400
    if [ ! -d "datasets/open_kits19/data" ]; then
        tar -zxf download/kits19.tar.gz -C datasets/open_kits19
    fi
}

# # GaitDatasetB-silh_GaitSet_preprocess
# if [ ! -f "download/GaitDatasetB-silh_GaitSet_preprocess.tar.gz" ]; then
#     $mc cp alias_name/solution/dataset/GaitDatasetB-silh_GaitSet_preprocess.tar.gz download/GaitDatasetB-silh_GaitSet_preprocess.tar.gz
# fi

# # oGaitDatasetB-silh_GaitSet_preprocess
# if [ ! -d "datasets/open_gait_b/data_pre" ]; then
#     tar -zxf download/GaitDatasetB-silh_GaitSet_preprocess.tar.gz -C datasets/open_gait_b
# fi


# # celeba.zip
# if [ ! -f "download/celeba.zip" ]; then
#     $mc cp alias_name/solution/raw_dataset/celeba.zip download/celeba.zip
# fi

# # celeba.zip
# if [ ! -d "datasets/open_celeba/data" ]; then
#     unzip download/celeba.zip -d datasets/open_celeba  > /dev/null
# fi

# # C3D_ucf101_sample.tar.gz
# if [ ! -f "download/C3D_ucf101_sample.tar.gz" ]; then
#     $mc cp alias_name/solution/dataset/C3D_ucf101_sample.tar.gz download/C3D_ucf101_sample.tar.gz
# fi

# # C3D_ucf101_sample.tar.gz
# if [ ! -d "datasets/open_ucf101/data/rawframes" ]; then
#     tar -zxf download/C3D_ucf101_sample.tar.gz -C datasets/open_ucf101/data
# fi

# # P3D_ucf101_sample.tar.gz
# if [ ! -f "download/P3D_ucf101_sample.tar.gz" ]; then
#     $mc cp alias_name/solution/dataset/P3D_ucf101_sample.tar.gz download/P3D_ucf101_sample.tar.gz
# fi

# # P3D_ucf101_sample.tar.gz
# if [ ! -d "datasets/open_ucf101/data/test" ]; then
#     tar -zxf download/P3D_ucf101_sample.tar.gz -C datasets/open_ucf101/data
# fi

# # ffhq_1024x1024.zip
# if [ ! -f "download/ffhq_1024x1024.zip" ]; then
#     $mc cp alias_name/solution/raw_dataset/ffhq_1024x1024.zip download/ffhq_1024x1024.zip
# fi

# # ffhq_1024x1024.zip
# if [ ! -d "datasets/open_ffhq_1024/ffhq_1024x1024" ]; then
#     unzip download/ffhq_1024x1024.zip -d datasets/open_ffhq_1024  > /dev/null
# fi


# download open imagenet
# if [ ! -x "venv/Tensorflow-v2.4.0" ]; then
#     $mc cp alias_name/solution/ci/venv.tar download/venv.tar
# fi

# Extracting venv
# if [ ! -x "venv/Tensorflow-v2.4.0" ]; then
#     tar xf download/venv.tar -C ./
# fi

function show_usage(){
    echo "Usage: ./$(basename $0) [OPTIONS]"
    echo "Options:"
    echo "    -d, --datasets='dataset_names ...'   set dataset_names"
    echo "    -h, --help                           show the usage"
    echo "example: ./prepare_open.sh"    
    echo "example: ./prepare_open.sh -d 'open_imagenet open_squad'"
}

function handle_params(){
    # if [[ -z $@ ]];then
    #     show_usage
    # fi

    while getopts ":d:h" opt; do
        case $opt in
            d ) G_DATASET_ARRAY=($OPTARG) ;;
            h ) show_usage; exit -1 ;;
        esac
    done
    shift $(( $OPTIND - 1 ))
    
    if [[ ${#G_DATASET_ARRAY[@]} -gt 0 ]]; then
        echo "the parameters value list"
        echo "G_DATASET_ARRAY:     ${G_DATASET_ARRAY[@]}"
    fi
}

function dataset_handler(){    
    minio_client_download

    local func_name=""
    if [[ ${#G_DATASET_ARRAY[@]} -eq 0 ]]; then
        G_DATASET_ARRAY=(${DATASET_ARRAY[@]})
    fi

    pid_array=()
    for d in ${G_DATASET_ARRAY[@]}; do
        func_name="download_extract_${d}"
        ( echo "$d ===> start ......" ;
          $func_name ;
          echo "$d ===> finished !!!" ; )&
        pid_array[$i]=$!
    done

    for pid in "${pid_array[@]}"; do
        wait $pid
    done


   echo "All datasets download and extract done."
}

handle_params "$@"
dataset_handler

