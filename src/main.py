import torch

'''
    python main.py -c configs/FOLO.json pretrain1
'''

def put_default_config(config):
    def set_default(key, value):
        if key not in config:
            config[key] = value

    set_default('master_addr', '127.0.0.1')
    import random
    set_default('master_port', random.randint(30000, 40000))  # This random goes before the pseudo random
    set_default('seed', 100)
    # set_default('root_dir', '/root/kg-datasets/KG_data') linux
    set_default('root_dir', 'F:/kg-datasets/KG_data')
    set_default('data_dir', f'{config["root_dir"]}/FOLO-q2b')

    # 训练参数
    set_default('num_epoch', 2)
    # AdamW SGD ASGD
    set_default('optimizer', 'AdamW') # Adam对Transformer的效果不好
    set_default('cache_ttl', 5)
    set_default('is_resume', False)
    set_default('upstream_task_name', None)
    set_default('downstream_task_name', 'default') # default # default 
    set_default('num_workers', 4)
    set_default('batch_size', 256) # 64
    set_default('epoch_choosing', False)
    set_default('grad_accum', 1)
    set_default('pre_norm', True)
    set_default('save_interval', 1) # 隔几个epoch保存一次
    set_default('test_interval', 3) # 隔几个epoch检测一次
    
    # 超参数
    set_default('hidden_size', 1024) 
    set_default('num_heads', 8)
    set_default('num_layers', 6)
    set_default('dim_feedforward', 1024) 
    set_default('lr', 1e-4) # 
    set_default('scheduler', 'exp')
    set_default('exponential_lr_rate', 0.99)
    set_default('loss', 'CE')
    set_default('smoothing', 0)
    set_default('eta_min', 0)
    set_default('mask_ratio', 0.8)
    set_default('p_mask_ratio', 1.0)
    set_default('dropout', 0.1)
    set_default('attention_dropout', 0.1)

    # 预训练采样参数
    set_default('pretrain_mask_ratio', [0.2, 0.4])  # BERT [mask] token ratio range
    set_default('pretrain_mask_type_ratio', [1, 0])  # Ratio of entity : relation
    set_default('pretrain_dataset_source', 'entity')  # 'relation' or 'entity'
    set_default('edge_drop_out_rate', 0)
    set_default('sample_retries', 1) # 5
    set_default('ladies_size', 8)
    set_default('pretrain_sampler_ratio', {
        '1p': 0,
        '2p': 0,
        # '3p': 0,
        # '2i': 0,
        # '3i': 0, 
        # 'meta_tree': 5,
        # 'ladies': 5,
    })
    set_default('induced_edge_prob', 0.8)
    
    # 推理项目
    set_default('reasoning_train_modes', ['1p', '2p']) # ['1p', '2p', '3p', '2i', '3i']
    set_default('reasoning_test_modes', ['pi']) # ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up']
    
    # Epoch choosing
    set_default('from_best', False)
    set_default('save_best', False)
    return config

def run_reasoning(config):
    from tasks.pretrain import FOLO
    pretrain_task = FOLO(config)
    from tasks.reasoning import reasoning
    downstream_task = reasoning(
        pretrain_task.betae,
        relation_cnt=pretrain_task.relation_cnt,
        # config['reasoning_train_modes']: ['1p', '2p', '3p', '2i', '3i'])
        train_mode=config['reasoning_train_modes'],
        # config['reasoning_test_modes']: ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up']
        test_mode=config['reasoning_test_modes'][:1], # 2u up does not support mixing 
    )
    if config['from_best']: # False
        best_input = config['upstream_task_name']
        best_result = 0
        valid_dl = downstream_task.dataloader_valid(config)
        for i in range(1, 10000000):
            from pathlib import Path
            task_name = f'{config["upstream_task_name"]}{i}'
            load_path = Path('pretrained_model/FOLO_camera_pretrain2.pt')
            if not load_path.exists():
                break
            from train import ft_test
            cur_val = ft_test(
                valid_dl,
                num_nodes=pretrain_task.num_nodes,
                relation_cnt=pretrain_task.relation_cnt,
                config=config,
                task_name=task_name,
                quiet=True,
            )
            if cur_val > best_result:
                best_input = task_name
                best_result = cur_val
        del valid_dl
        config['upstream_task_name'] = best_input
        torch.cuda.empty_cache()

    # Finetune training
    print(100*'*' + '\nFinetune training time!\n' + 100*'*')
    from train import main_mp

    test_modes = config['reasoning_test_modes']
    if test_modes is None or test_modes == []:
        test_loader = None
    else: # 执行
        test_loader = downstream_task.dataloader_test(config) # 读取test的pkl

    '''Train'''
    # print('MAIN 145',len(downstream_task.dataloader_fine_tune(config))) 
    main_mp(
        config,
        pretrain_task.num_nodes, # 0
        pretrain_task.relation_cnt, # 0
        downstream_task.dataloader_fine_tune(config), 
        test_loader,
        valid_loader=downstream_task.dataloader_valid(config) if config['save_best'] else None,
    )

    del pretrain_task
    del downstream_task
    if test_loader is not None:
        run_test_reasoning(config)

def run_test_reasoning(config):
    from tasks.pretrain import FOLO
    from tasks.reasoning import reasoning
    pretrain_task = FOLO(config)
    test_modes = config['reasoning_test_modes']

    from train import ft_test
    for mode in test_modes:
        print('Testing mode', mode)
        downstream_task = reasoning(
            pretrain_task.betae,
            relation_cnt=pretrain_task.relation_cnt,
            train_mode=[],
            test_mode=[mode],
        )
        best_input = config['downstream_task_name']
        
        if config['from_best']: # False
            best_result = 0
            valid_dl = downstream_task.dataloader_valid(config)
            for i in range(1, 3): # 10000000
                from pathlib import Path
                task_name = f'{config["downstream_task_name"]}{i}'
                load_path = f'{task_name}.pt'
                if not load_path.exists():
                    break
                from train import ft_test
                cur_val = ft_test(
                    valid_dl,
                    num_nodes=pretrain_task.num_nodes,
                    relation_cnt=pretrain_task.relation_cnt,
                    config=config,
                    task_name=task_name,
                    quiet=True,
                )
                if cur_val > best_result:
                    best_input = task_name
                    best_result = cur_val
            del valid_dl
        cjk_config = (pretrain_task.betae,pretrain_task.relation_cnt,[],[mode])
        ft_test(            
            cjk_config,
            downstream_task.dataloader_test(config),
            num_nodes=pretrain_task.num_nodes,
            relation_cnt=pretrain_task.relation_cnt,
            config=config,
            task_name=best_input
        )

def get_argparser():
    import argparse
    parser = argparse.ArgumentParser(prog='python main.py', description='KGTransformer')    
    parser.add_argument('-c', '--config', default='configs/drugbank.json', nargs=1, type=argparse.FileType('r'), help='path to the config file')

    return parser

def dfs_parsing(config_list, parse_status, task):
    stat = parse_status.get(task)
    if stat == 'Done':
        return
    if stat == 'Parsing':
        assert False, f'Loop detected in config.'
    parse_status[task] = 'Parsing'
    if task not in config_list:
        assert False, f'Task {task} not found'
    config = config_list[task]
    if 'base' in config:
        dfs_parsing(config_list, parse_status, config['base'])
        config_base = config_list[config['base']]
        del config['base']
        for k in config_base:
            if k not in config:
                config[k] = config_base[k]
    put_default_config(config)
    parse_status[task] = 'Done'

def args_to_config(args):
    import json
    config_list = json.load(args.config)
    assert isinstance(config_list, dict), "Config should be an dict of tasks."
    parse_status = dict()
    for task in config_list:
        dfs_parsing(config_list, parse_status, task)
    return config_list

def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # task = 'reasoning'
    task = 'test_pretrain'
    print('Running KGTransformer')
    args = get_argparser().parse_args()
    # print('Arguments:', args)
    config_list = args_to_config(args)
    t = task
    if t not in config_list:
        assert False, f'Task {t} not found in config'

    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    torch.set_printoptions(profile='full')

    config = config_list[t]
    print(f'Running task "{t}". Definitive config:')
    import json
    print(json.dumps(config))
    set_seed(config['seed'])
    # Environments
    import os
    os.environ['MASTER_ADDR'] = config['master_addr']
    os.environ['MASTER_PORT'] = str(config['master_port'])

    torch.cuda.empty_cache()
    if config['type'] == 'reasoning':
        run_reasoning(config)
    elif config['type'] == 'test-reasoning':
        run_test_reasoning(config)
    else:
        assert False, f'Task {t} is not runnable.'

# python main.py -c configs/FOLO.json pretrain1
if __name__ == '__main__':
    main()
