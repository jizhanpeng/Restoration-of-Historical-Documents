import os
import torchvision.models
from schedule.schedule import Schedule
from model.DocDiff import DocDiff
from schedule.resshift_sample_forword import *
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import copy
from src.sobel import Laplacian
from utils.loss_sty import *
from data.docdata1 import *
from model.NAFNet import *
from utils.psnr import *
from torch.cuda.amp import autocast, GradScaler
import lpips


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16()

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]

import torch.distributed as dist
import os

def init__result_Dir():
    work_dir = os.path.join(os.getcwd(), 'Training')
    path = None

    if dist.get_rank() == 0:  # 只有主进程计算并创建目录
        max_model = 0
        for root, j, file in os.walk(work_dir):
            for dirs in j:
                try:
                    temp = int(dirs)
                    if temp > max_model:
                        max_model = temp
                except:
                    continue
            break
        max_model += 1
        path = os.path.join(work_dir, str(max_model))
        os.makedirs(path, exist_ok=True)  # 使用 exist_ok=True 避免报错
        print(f"[Rank 0] 创建结果目录: {path}")

    # 同步：让其他进程等待 rank 0 创建完成
    dist.barrier()

    # 所有进程：重新获取最新目录（rank 0 创建的那个）
    if dist.get_rank() != 0:
        dirs = sorted(
            [int(d) for d in os.listdir(work_dir) if d.isdigit()],
            reverse=True
        )
        if dirs:
            path = os.path.join(work_dir, str(dirs[0]))
        else:
            raise RuntimeError("Rank 0 未创建训练目录！")

    return path


class Trainer:
    def __init__(self, config):
        self.rank = config.RANK
        self.world_size = config.WORLD_SIZE
        self.NAFcriterion = PSNRLoss().cuda()
        self.criterion = CPLoss()
        self.vgg = VGG16().cuda()
        self.mode = config.MODE
        self.schedule = Schedule(config.SCHEDULE, config.TIMESTEPS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = config.CHANNEL_X
        out_channels = config.CHANNEL_Y
        self.out_channels = out_channels
        #self.process = NAFNet().cuda()
        from uformer_model import Uformer
        self.process = Uformer(img_size=512,embed_dim=16,win_size=8,token_projection='linear',token_mlp='leff',modulator=True)
        self.network = DocDiff(
            input_channels=in_channels,
            output_channels=out_channels,
            n_channels=config.MODEL_CHANNELS,
            ch_mults=config.CHANNEL_MULT,
            n_blocks=config.NUM_RESBLOCKS
        ).to(self.device)
        self.diffusion = create_gaussian_diffusion(denoiser=self.network.denoiser)
        self.test_init_predict_save_path = config.test_init_predict_save_path
        if not os.path.exists(self.test_init_predict_save_path):
            os.makedirs(self.test_init_predict_save_path)
        self.test_sampledImgs_save_path = config.test_sampledImgs_save_path
        if not os.path.exists(self.test_sampledImgs_save_path):
            os.makedirs(self.test_sampledImgs_save_path)
        self.test_finalImgs_save_path = config.test_finalImgs_save_path
        if not os.path.exists(self.test_finalImgs_save_path):
            os.makedirs(self.test_finalImgs_save_path)
        self.pretrained_path_init_predictor = config.PRETRAINED_PATH_INITIAL_PREDICTOR
        self.pretrained_path_denoiser = config.PRETRAINED_PATH_DENOISER
        self.continue_training = config.CONTINUE_TRAINING
        self.continue_training_steps = 0
        self.path_train_damaged = config.PATH_DAMAGED
        self.path_train_mask = config.PATH_MASK
        self.path_train_content = config.PATH_CONTENT
        self.path_train_gt = config.PATH_GT
        self.iteration_max = config.ITERATION_MAX
        self.LR = config.LR
        self.cross_entropy = nn.BCELoss()
        self.num_timesteps = config.TIMESTEPS
        self.save_model_every = config.SAVE_MODEL_EVERY
        self.weight_save_path = config.WEIGHT_SAVE_PATH
        self.TEST_INITIAL_PREDICTOR_WEIGHT_PATH = config.TEST_INITIAL_PREDICTOR_WEIGHT_PATH
        self.TEST_DENOISER_WEIGHT_PATH = config.TEST_DENOISER_WEIGHT_PATH
        self.DPM_STEP = config.DPM_STEP
        self.test_path_img = config.TEST_PATH_IMG
        self.test_path_gt = config.TEST_PATH_GT
        self.beta_loss = config.BETA_LOSS
        self.pre_ori = config.PRE_ORI
        self.high_low_freq = config.HIGH_LOW_FREQ
        self.image_size = config.IMAGE_SIZE
        self.native_resolution = config.NATIVE_RESOLUTION
        lpips_net = 'vgg'
        lpips_loss = lpips.LPIPS(net=lpips_net).to(self.device)

        lpips_loss.scaling_layer.shift = lpips_loss.scaling_layer.shift.to(self.device)
        lpips_loss.scaling_layer.scale = lpips_loss.scaling_layer.scale.to(self.device)

        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        self.lpips_loss = lpips_loss
        if self.mode == 1 and self.continue_training == 'True':
            print('Continue Training')
            state_dict_init = torch.load(self.pretrained_path_init_predictor)
            if hasattr(self.process, 'module'):
                self.process.module.load_state_dict(state_dict_init)
            else:
                self.process.load_state_dict(state_dict_init)

            state_dict_denoiser = torch.load(self.pretrained_path_denoiser)
            if hasattr(self.network, 'module'):
                self.network.module.denoiser.load_state_dict(state_dict_denoiser)
            else:
                self.network.denoiser.load_state_dict(state_dict_denoiser)
            self.continue_training_steps = config.CONTINUE_TRAINING_STEPS
        if self.mode == 1:
            dataset_train = TrainDataset(config)
            self.batch_size = config.BATCH_SIZE

            # ✅ 添加 DistributedSampler
            self.sampler_train = torch.utils.data.distributed.DistributedSampler(
                dataset_train,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )

            self.dataloader_train = DataLoader(
                dataset_train,
                batch_size=self.batch_size,
                shuffle=False,  # ⚠️ 必须设为 False，sampler 负责 shuffle
                drop_last=True,  # 建议 True，避免 batch size 不一致
                num_workers=config.NUM_WORKERS,
                prefetch_factor=2,
                sampler=self.sampler_train  # ✅ 关键！
            )
        else:
            dataset_test = PaperDamageTestDataset(config)
            self.dataloader_test = DataLoader(dataset_test, batch_size=config.BATCH_SIZE_VAL, shuffle=False,
                                              drop_last=False,
                                              num_workers=config.NUM_WORKERS)
        if config.LOSS == 'L1':
            self.loss = nn.L1Loss()
        elif config.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')
            self.loss = nn.MSELoss()
        if self.high_low_freq == 'True':
            self.high_filter = Laplacian().to(self.device)

        # 在 __init__ 最后添加（假设你已传入 rank 和 world_size）

        # 初始化进程组（如果尚未初始化）
        if not dist.is_initialized():
            setup(self.rank, self.world_size)

        # 将模型移动到当前 GPU
        torch.cuda.set_device(self.rank)
        self.process = self.process.to(self.rank)
        self.network = self.network.to(self.rank)
        self.vgg = self.vgg.to(self.rank)

        # 用 DDP 包装模型
        self.process = torch.nn.parallel.DistributedDataParallel(self.process, device_ids=[self.rank])
        self.network = torch.nn.parallel.DistributedDataParallel(self.network, device_ids=[self.rank])

        # --- ✅ 新增：初始化 EMA ---
        self.ema_decay = 0.9999  # 推荐 0.9999
        self.ema_update_every = 1  # 每几步更新一次

        # 创建 EMA 模型副本（不参与梯度计算）
        self.ema_process = copy.deepcopy(self.process.module if hasattr(self.process, 'module') else self.process).eval().to(self.rank)
        self.ema_denoiser = copy.deepcopy(self.network.module.denoiser if hasattr(self.network, 'module') else self.network.denoiser).eval().to(self.rank)

        # 冻结 EMA 模型参数
        for param in self.ema_process.parameters():
            param.requires_grad = False
        for param in self.ema_denoiser.parameters():
            param.requires_grad = False

        # 初始化计数器
        self.ema_step = 0

    def test(self):




    def train(self):


