###Usage:
1. Environment: Python 3.8,torch-1.4.0+cu100
2. To train the meta\_CF models, cd the Meta\_CF directory and execute the command'python main.py --mod=train --model=<model\_name> --dataset=<dataset\_name> --local\_lr=<local\_update\_learning\_rate>  --global_lr=<global\_update\_learning\_rate> --regs=<regularization> --topK=<topK\_used\_when\_evaluation>
3. To test the model trained, use the same parameters as training except the mod replaced by 'test'

Following this training example:'python main.py --mod=train --model=model3\_linear --dataset=amazon --local\_lr=0.01 --global\_lr=0.00001 --regs=0.01'

Following this test example corresponding to the training example:'python test.py --mod=test --model=model3\_linear --dataset=amazon --local\_lr=0.01 --global\_lr=0.00001 --regs=0.01' 
