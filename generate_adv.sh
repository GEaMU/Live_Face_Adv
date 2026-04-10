# the --num option value need to be set less than or equal to your src images numbers.
python eval3D.py --model IR152 --dataset 3D --t 999 --save res3D_IR152Bigo --num 701
python eval3D.py --model IR152 --dataset 3D --t 999 --save res3D_IR152o --num 1300
python eval3D.py --model IR152 --dataset celeba --t 999 --save resceleba_IR152 --num 1000

#python -m pytorch_fid /data/Adv-Diffusion-main/3D/src /data/Adv-Diffusion-main/res3D_IRSE50oBig2025/img
#python psnr_ssim.py --dir0 /data/Adv-Diffusion-main/3D/src --dir1 /data/Adv-Diffusion-main/res3D_IRSE50oBig2025/img
