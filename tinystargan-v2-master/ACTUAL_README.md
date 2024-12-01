python main.py \
--mode sample \
--num_domains 2 \
--w_hpf 1 \
--alpha 64 \
--efficient 1 \
--resume_iter 250000 \
--checkpoint_dir expr/checkpoints/ \
--val_batch_size 16 \
--latent_sample_per_domain 1024 \
--filename tiny_celeb_male.jpg \
--src_dir assets/representative/celeb_male/src \
--ref_dir assets/representative/celeb_male/ref 

Change --latent_sample_per_domain to larger number to generate more images


# Run FID
python -m pytorch_fid --dims 2048 "./expr/results" "./data/celeb_male/train/male"