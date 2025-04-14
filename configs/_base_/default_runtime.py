weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 16  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = None  # auto adapt to bs 1 for each gpu
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch
clip_grad = None  # disable with None, enable with a float

sync_bn = False
enable_amp = False
amp_dtype = "float16"
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False

enable_wandb = True
wandb_project = "pointcept"  # custom your project name e.g. Sonata, PTv3
wandb_key = None  # wandb token, default is None. If None, login with `wandb login` in your terminal

mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# Trainer
train = dict(type="DefaultTrainer")

# Tester
test = dict(type="SemSegTester", verbose=True)
