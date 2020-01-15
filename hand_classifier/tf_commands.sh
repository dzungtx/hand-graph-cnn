make_image_classifier \
  --image_dir data/images \
  --tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 \
  --image_size 224 \
  --saved_model_dir weights \
  --labels_output_file class_labels.txt \
  --tflite_output_file new_mobile_model.tflite

python label_image.py \
  --input_mean 0 --input_std 255 \
  --model_file new_mobile_model.tflite --label_file class_labels.txt \
  --image ../data/random/images/