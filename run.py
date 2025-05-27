import os
import sys
import time
import random
import argparse
import yaml
import Retrieval


from utils.hdfs_io import HADOOP_BIN, hexists, hmkdir, hcopy

def get_dist_launch(args):  # some examples
    
    if args.dist == 'single':  #单卡调试模式
        return "CUDA_VISIBLE_DEVICES=3 python -W ignore"

    elif args.dist == 'f2':
        return "CUDA_VISIBLE_DEVICES=3,4 WORLD_SIZE=2 /home/16T/miniconda/zzl/envs/zzl3.9/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=2 " \
               "--nnodes=1 "

    elif args.dist.startswith('gpu'):  # use one gpu, --dist "gpu0"
        num = int(args.dist[3:])
        assert 0 <= num <= 8
        return "CUDA_VISIBLE_DEVICES={:} WORLD_SIZE=1 /home/16T/miniconda/zzl/envs/zzl3.9/bin/python -W ignore -m torch.distributed.launch --master_port 9999 --nproc_per_node=1 " \
               "--nnodes=1 ".format(num)

    else:
        raise ValueError


def get_from_hdfs(file_hdfs):
    """
    compatible to HDFS path or local path
    """
    if file_hdfs.startswith('hdfs'):
        file_local = os.path.split(file_hdfs)[-1]

        if os.path.exists(file_local):
            print(f"rm existing {file_local}")
            os.system(f"rm {file_local}")

        hcopy(file_hdfs, file_local)

    else:
        file_local = file_hdfs
        assert os.path.exists(file_local)

    return file_local

def run_retrieval(args):
    #单卡模式，方便调试
    if args.dist == 'single':
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)  
        
        args.device = "cuda:4"  #显卡编号
        args.distributed = False

        Retrieval.main(args, config)

    else:
        dist_launch = get_dist_launch(args)
        use_env_flag = "--use_env" if "torch.distributed.launch" in dist_launch else ""
        os.system(f"{dist_launch} "
                  f"{use_env_flag} Retrieval.py --config {args.config} "
                  f"--output_dir {args.output_dir} --bs {args.bs} --checkpoint {args.checkpoint} {'--evaluate' if args.evaluate else ''}")

def run(args):
    if args.task == 'itr_rsicd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsicd")
        args.config = 'configs/Retrieval_rsicd_vit.yaml'
        run_retrieval(args)

    elif args.task == 'itr_rsitmd_vit':
        # assert os.path.exists("../X-VLM-pytorch/images/rsitmd")
        args.config = 'configs/Retrieval_rsitmd_vit.yaml'
        run_retrieval(args)
    else:
        raise NotImplementedError(f"task == {args.task}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='itr_rsicd_vit')
    parser.add_argument('--dist', type=str, default='single', help="see func get_dist_launch for details")
    parser.add_argument('--config', default='configs/Retrieval_rsicd_vit.yaml', type=str, help="if not given, use default")
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus; "
                                                  "this option only works for fine-tuning scripts.")
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--checkpoint', default='-1', type=str, help="for fine-tuning")
    parser.add_argument('--load_ckpt_from', default=' ', type=str, help="load domain pre-trained params")
    # write path: local or HDFS
    parser.add_argument('--output_dir', type=str, default='./checkpoints/full_rsicd_vit', help='for fine-tuning, local path; '
                                                                      'for pre-training, local and HDFS are both allowed.')
    parser.add_argument('--evaluate', action='store_true', default=False, help="evaluation on downstream tasks")
    args = parser.parse_args()
    assert hexists(os.path.dirname(args.output_dir))
    hmkdir(args.output_dir)
    run(args)

