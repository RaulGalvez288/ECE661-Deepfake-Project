python main.py --mode train --num_domains 2 --w_hpf 1 \
               --alpha 64 --efficient 1 \
               --lambda_reg 1 --lambda_sty 1 --lambda_ds 1 --lambda_cyc 1 \
               --train_img_dir data/celeb_male/train \
               --val_img_dir data/celeb_male/val \
               --checkpoint_dir expr/checkpoints/ \
               --num_workers 0 \
               --save_every 10000 \
               --resume_iter 300000 \
               --total_iters 400000



python main.py \
--mode sample \
--num_domains 2 \
--w_hpf 1 \
--alpha 64 \
--efficient 1 \
--resume_iter 300000 \
--checkpoint_dir expr/checkpoints/ \
--val_batch_size 16 \
--latent_sample_per_domain 1024 \
--filename tiny_celeb_male.jpg \
--src_dir assets/representative/celeb_male/src \
--ref_dir assets/representative/celeb_male/ref \
--result_dir expr/results_300/



python main.py \
--mode sample \
--num_domains 2 \
--w_hpf 1 \
--alpha 64 \
--efficient 1 \
--resume_iter 300000 \
--checkpoint_dir expr/checkpoints/ \
--val_batch_size 16 \
--latent_sample_per_domain 500 \
--filename tiny_celeb_male.jpg \
--src_dir assets/representative/celeb_male/src \
--ref_dir assets/representative/celeb_male/ref \
--result_dir expr/results_300/

Change the corresponding checkpoint file in core/checkpoint.py since it is hardcoded
Change --latent_sample_per_domain to larger number to generate more images


# Run FID
ls ./data/celeb_male/train/male/* | head -n 10 | xargs -I {} cp {} test_real/
ls ./expr/results/* | head -n 3000 | xargs -I {} cp {} test_fake/
python -m pytorch_fid --dims 2048 "./expr/results" "./data/celeb_male/train/male"


python main.py \
--mode sample \
--num_domains 2 \
--w_hpf 1 \
--alpha 64 \
--efficient 1 \
--resume_iter 300000 \
--checkpoint_dir expr/checkpoints/ \
--val_batch_size 16 \
--latent_sample_per_domain 500 \
--filename tiny_celeb_male.jpg \
--src_dir assets/representative/celeb_male/src \
--ref_dir assets/representative/celeb_male/ref \
--result_dir expr/results_300/