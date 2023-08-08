# train cifar10
# train reference model
python main.py --id=0 -g=0
# train unrelated model with different random seed
python main.py --id=u1 -s=1 -g=0
python main.py --id=u2 -s=2 -g=0
python main.py --id=u3 -s=3 -g=0
python main.py --id=u4 -s=4 -g=0
python main.py --id=u5 -s=5 -g=0
# train directly unlearning model with different random seed
python main.py --id=dir1 -s=1 --ul -g=0
python main.py --id=dir2 -s=2 --ul -g=0
python main.py --id=dir3 -s=3 --ul -g=0
python main.py --id=dir4 -s=4 --ul -g=0
python main.py --id=dir5 -s=5 --ul -g=0
# train approximate unlearning model with different random seed
python main.py --model-dir=models/ckpt_CIFAR10_0/model_epoch_200 --mix-label --id=app1 -s=1 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR10_0/model_epoch_200 --mix-label --id=app2 -s=2 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR10_0/model_epoch_200 --mix-label --id=app3 -s=3 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR10_0/model_epoch_200 --mix-label --id=app4 -s=4 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR10_0/model_epoch_200 --mix-label --id=app5 -s=5 --ul -g=0
# train cifar100
# train reference model
python main.py --dataset=CIFAR100 --nc=100 --id=0 -g=0
# train unrelated model with different random seed
python main.py --dataset=CIFAR100 --nc=100 --id=u1 -s=1 -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=u2 -s=2 -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=u3 -s=3 -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=u4 -s=4 -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=u5 -s=5 -g=0
# train directly unlearning model with different random seed
python main.py --dataset=CIFAR100 --nc=100 --id=dir1 -s=1 --ul -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=dir2 -s=2 --ul -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=dir3 -s=3 --ul -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=dir4 -s=4 --ul -g=0
python main.py --dataset=CIFAR100 --nc=100 --id=dir5 -s=5 --ul -g=0
# train approximate unlearning model with different random seed
python main.py --model-dir=models/ckpt_CIFAR100_0/model_epoch_200 --mix-label --dataset=CIFAR100 --nc=100 --id=app1 -s=1 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR100_0/model_epoch_200 --mix-label --dataset=CIFAR100 --nc=100 --id=app2 -s=2 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR100_0/model_epoch_200 --mix-label --dataset=CIFAR100 --nc=100 --id=app3 -s=3 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR100_0/model_epoch_200 --mix-label --dataset=CIFAR100 --nc=100 --id=app4 -s=4 --ul -g=0
python main.py --model-dir=models/ckpt_CIFAR100_0/model_epoch_200 --mix-label --dataset=CIFAR100 --nc=100 --id=app5 -s=5 --ul -g=0
# reference model vs unrelated model, cifar10
python compare_field.py --rs=cf10_ref_vs_un1 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_u1 -g=0
python compare_field.py --rs=cf10_ref_vs_un2 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_u2 -g=0
python compare_field.py --rs=cf10_ref_vs_un3 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_u3 -g=0
python compare_field.py --rs=cf10_ref_vs_un4 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_u4 -g=0
python compare_field.py --rs=cf10_ref_vs_un5 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_u5 -g=0
# reference model vs unrelated model, cifar100
python compare_field.py --nc=100 --rs=cf100_ref_vs_un1 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_u1 -g=0
python compare_field.py --nc=100 --rs=cf100_ref_vs_un2 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_u2 -g=0
python compare_field.py --nc=100 --rs=cf100_ref_vs_un3 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_u3 -g=0
python compare_field.py --nc=100 --rs=cf100_ref_vs_un4 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_u4 -g=0
python compare_field.py --nc=100 --rs=cf100_ref_vs_un5 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_u5 -g=0
# reference model vs directly unlearning model, cifar10
python compare_field.py --rs=cf10_ref_vs_dir1 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_dir1 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_dir2 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_dir2 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_dir3 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_dir3 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_dir4 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_dir4 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_dir5 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_dir5 --fix-epoch=200 -g=0
# reference model vs directly unlearning model, cifar100
python compare_field.py --rs=cf100_ref_vs_dir1 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_dir1 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_dir2 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_dir2 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_dir3 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_dir3 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_dir4 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_dir4 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_dir5 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_dir5 --fix-epoch=200 -g=0
# reference model vs approximate unlearning model, cifar10
python compare_field.py --rs=cf10_ref_vs_app1 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_app1 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_app2 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_app2 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_app3 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_app3 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_app4 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_app4 --fix-epoch=200 -g=0
python compare_field.py --rs=cf10_ref_vs_app5 --path1=ckpt_CIFAR10_0 --path2=ckpt_CIFAR10_app5 --fix-epoch=200 -g=0
# reference model vs approximate unlearning model, cifar10
python compare_field.py --rs=cf100_ref_vs_app1 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_app1 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_app2 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_app2 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_app3 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_app3 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_app4 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_app4 --fix-epoch=200 -g=0
python compare_field.py --rs=cf100_ref_vs_app5 --nc=100 --path1=ckpt_CIFAR100_0 --path2=ckpt_CIFAR100_app5 --fix-epoch=200 -g=0

# plot
python plot.py