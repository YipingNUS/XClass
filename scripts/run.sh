export PYTHONIOENCODING=utf8

set -e

gpu=$1
dataset=$2
echo "EXECUTING: static_representations.py"
CUDA_VISIBLE_DEVICES=${gpu} python3 static_representations.py --dataset_name ${dataset}

echo "EXECUTING: class_oriented_document_representations.py"
CUDA_VISIBLE_DEVICES=${gpu} python3 class_oriented_document_representations.py --dataset_name ${dataset}

echo "EXECUTING: document_class_alignment.py"
python3 document_class_alignment.py --dataset_name ${dataset} --pca 64

echo "EXECUTING: evaluate.py --stage Rep --suffix bbu-12-mixture-100"
python3 evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100

echo "EXECUTING: evaluate.py --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42"
python3 evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42

echo "EXECUTING: prepare_text_classifer_training.py"
python3 prepare_text_classifer_training.py --dataset_name ${dataset} --suffix pca64.clusgmm.bbu-12.mixture-100.42

echo "EXECUTING: run_train_text_classifier.sh"
./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.bbu-12.mixture-100.42.0.5

echo "EXECUTING: evaluate.py --stage Rep --suffix bbu-12-mixture-100"
python3 evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100

echo "EXECUTING: evaluate.py --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42"
python3 evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42
