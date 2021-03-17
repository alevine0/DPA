python3 main_supcon.py --batch_size 256 \
  --learning_rate 0.5 \
  --temp 0.5 \
  --model resnet18 \
  --cosine \
  --num_workers 1 \
  --trial 1 \
  --size 48 \
  --dataset gtsrb \
  --method SimCLR