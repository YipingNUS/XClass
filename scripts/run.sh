set -e

gpu=$1
dataset=$2
CUDA_VISIBLE_DEVICES=${gpu} python3 static_representations.py --dataset_name ${dataset}
CUDA_VISIBLE_DEVICES=${gpu} python3 class_oriented_document_representations.py --dataset_name ${dataset}
python3 document_class_alignment.py --dataset_name ${dataset}
python3 evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100
python3 evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42
python3 prepare_text_classifer_training.py --dataset_name ${dataset}
./run_train_text_classifier.sh ${gpu} ${dataset} pca64.clusgmm.bbu-12.mixture-100.42.0.5
python3 evaluate.py --dataset ${dataset} --stage Rep --suffix bbu-12-mixture-100
python3 evaluate.py --dataset ${dataset} --stage Align --suffix pca64.clusgmm.bbu-12.mixture-100.42
