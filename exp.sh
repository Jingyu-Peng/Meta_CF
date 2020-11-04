#CUDA_VISIBLE_DEVICES=2 python main.py --mod=exp --model=model1 --dataset=movielens --local_lr=0.005 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=exp --model=model2_linear --dataset=movielens --local_lr=0.005 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=exp --model=model2_neural --dataset=movielens --local_lr=0.005 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=exp --model=model3_linear --dataset=movielens --local_lr=0.005 &
#CUDA_VISIBLE_DEVICES=2 python main.py --mod=exp --model=model3_neural --dataset=movielens --local_lr=0.005 &
CUDA_VISIBLE_DEVICES=3 python main.py --mod=exp --model=model1 --dataset=amazon --local_lr=0.01 --regs=0.01 &
CUDA_VISIBLE_DEVICES=3 python main.py --mod=exp --model=model2_linear --dataset=amazon --local_lr=0.01 --regs=0.01 &
CUDA_VISIBLE_DEVICES=3 python main.py --mod=exp --model=model2_neural --dataset=amazon --local_lr=0.01 --regs=0.01 &
CUDA_VISIBLE_DEVICES=3 python main.py --mod=exp --model=model3_linear --dataset=amazon --local_lr=0.01 --regs=0.01 &
CUDA_VISIBLE_DEVICES=3 python main.py --mod=exp --model=model3_neural --dataset=amazon --local_lr=0.01 --regs=0.01 &
