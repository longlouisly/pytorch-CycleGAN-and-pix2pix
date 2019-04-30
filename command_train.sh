python train.py --dataroot ./datasets/city/reconstruction/ --name aerial_rec_wgan_layer_norm_1_gpu --model pix2pix --direction AtoB --netG unet_128 --netD n_layers --n_layers_D 4 --gan_mode wgangp --gpu_ids 0 --continue_train --epoch_count 16 --batch_size 16

python train.py --dataroot ./datasets/circles/reconstruction/ --name circles_rec_5x --model pix2pix --direction AtoB --netG unet_128 --netD n_layers --n_layers_D 4 --gan_mode wgangp --gpu_ids 0 --batch_size 16

python train.py --dataroot ./datasets/city/reconstruction/ --name aerial_rec_5x --model pix2pix --direction AtoB --netG unet_128 --netD n_layers --n_layers_D 4 --gan_mode wgangp --gpu_ids 0 --batch_size 16
