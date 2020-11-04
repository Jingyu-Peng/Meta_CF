#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model1 --dataset=movielens --local_lr=0.005 --global_lr=0.0000001 --local_epoch=1 &
#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model2_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=1 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model2_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --local_epoch=1 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model3_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=1 &
#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model3_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --global_lr_share=0.00001  --local_epoch=1 &

#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model1 --dataset=movielens --local_lr=0.005 --global_lr=0.0000001 --local_epoch=2 &
#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model2_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=2 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model2_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --local_epoch=2 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model3_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=2 &
#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model3_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --global_lr_share=0.00001 --local_epoch=2 &

#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model1 --dataset=movielens --local_lr=0.005 --global_lr=0.0000001 --local_epoch=3 &
#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model2_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=3 &
#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model2_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --local_epoch=3 &
#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model3_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=3 &
#CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model3_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --global_lr_share=0.00001 --local_epoch=3 &

#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model1 --dataset=movielens --local_lr=0.005 --global_lr=0.0000001 --local_epoch=4 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model2_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=4 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model2_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --local_epoch=4 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model3_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 --local_epoch=4 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model3_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --global_lr_share=0.00001 --local_epoch=4 &




#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model1 --dataset=movielens --local_lr=0.005 --global_lr=0.0000001 &
#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model2_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 &
#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model2_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 &
#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model3_linear --dataset=movielens --local_lr=0.005 --global_lr=0.00001 &
#CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model3_neural --dataset=movielens --local_lr=0.005 --global_lr=0.000001 --global_lr_share=0.00001 &





CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model1 --dataset=amazon --local_lr=0.01 --global_lr=0.00001 --regs=0.01&
CUDA_VISIBLE_DEVICES=2 python main.py --mod=test --model=model2_linear --dataset=amazon --local_lr=0.01 --global_lr=0.00001 --regs=0.01 &
CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model2_neural --dataset=amazon --local_lr=0.01 --global_lr=0.0000001 --regs=0.01 &
CUDA_VISIBLE_DEVICES=1 python main.py --mod=test --model=model3_linear --dataset=amazon --local_lr=0.01 --global_lr=0.00001 --regs=0.01 &
CUDA_VISIBLE_DEVICES=3 python main.py --mod=test --model=model3_neural --dataset=amazon --local_lr=0.01 --global_lr=0.000001 --global_lr_share=0.00001 --regs=0.01 &
