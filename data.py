from datasets import load_dataset

ds = load_dataset("FreedomIntelligence/MileBench", cache_dir="data/")

import argparse
import logging
import torch
import sys
import time

# 配置日志记录
logging.basicConfig(filename='results.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class KvModeWorker:
    def __init__(self, model, kv_mode):
        self.model = model
        self.kv_mode = kv_mode
        sys.path.insert(0, '/home/xhpei/projects/LOOK-M/LLaVA-mix_merge_v1')
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.model.kv_token_merge.modify_llama import H2OLlamaAttention_drop, \
                                                            WeightedLlamaAttention_drop, \
                                                            PivotMergeLlamaAttention_drop, \
                                                            TextH2OLlamaAttention_drop, \
                                                            TextWeightedLlamaAttention_drop, \
                                                            TextPivotLlamaAttention_drop, \
                                                            PoolingWindowLlamaAttention_drop, \
                                                            AVGMergeLlamaAttention_drop, \
                                                            MeanH2OLlamaAttention_drop, \
                                                            TextAVGMergeLlamaAttention_drop, \
                                                            New_drop
        self.TAGET_MODULE = {
            "origin": None,
            "h2o": H2OLlamaAttention_drop,
            "weighted_merge": WeightedLlamaAttention_drop,
            "pivot_merge": PivotMergeLlamaAttention_drop,
            "text_prior_h2o": TextH2OLlamaAttention_drop,
            "text_prior_weighted_merge": TextWeightedLlamaAttention_drop,
            "text_prior_pivot_merge": TextPivotLlamaAttention_drop,
            "snapkv": PoolingWindowLlamaAttention_drop,
            "avg_merge": AVGMergeLlamaAttention_drop,
            "mean_h2o": MeanH2OLlamaAttention_drop,
            "text_prior_avg_merge": TextAVGMergeLlamaAttention_drop,
            "new": New_drop,
        }
        self._replace_forward()

    def _replace_forward(self):
        target_module = self.TAGET_MODULE.get(self.kv_mode)
        if target_module is None:
            # 如果当前 kv_mode 不需要特殊处理，直接返回
            return

        # 遍历模型的所有模块，替换与kv_mode匹配的forward方法
        for name, m in self.model.named_modules():
            if isinstance(m, target_module):
                m.kv_mode = self.kv_mode  # 假设模块有kv_mode属性

    def prepare(self, device):
        # 将模型移动到指定设备，并设置为eval模式
        self.model.to(device)
        self.model.eval()

    def forward(self, input_ids, **gen_kwargs):
        with torch.autocast(device_type=self.model.device.type, dtype=torch.bfloat16):
            return self.model.generate(input_ids, use_cache=True, **gen_kwargs)

    def clean_cache(self):
        # 清理缓存的方法
        target_module = self.TAGET_MODULE.get(self.kv_mode)
        if target_module is None:
            return
        for name, m in self.model.named_modules():
            if isinstance(m, target_module):
                m._clean_cache()


def speed_test(worker, warmup_iters=1, n_iters=2, batch_size=1, text_tokens=77, image_shape=(3, 224, 224), device='cuda'):
    # 设置设备
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this machine.")
    elif device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine.")
    elif device == 'cpu':
        device = torch.device('cpu')
    elif device == 'cuda':
        device = torch.device('cuda')
    elif device == 'mps':
        device = torch.device('mps')
    else:
        raise ValueError(f"Unsupported device: {device}")

    # 使用worker准备模型
    worker.prepare(device)

    # 准备输入数据
    input_ids = torch.randint(low=0, high=30522, size=(batch_size, text_tokens)).to(device)
    image_tensor = torch.rand((batch_size, *image_shape)).to(device)

    with torch.no_grad():
        # 预热迭代
        for _ in range(warmup_iters):
            print("c1")
            output_ids = worker.forward(input_ids)
            print("c1")
        logger.info('Start measuring speed.')

        if device == 'cuda':
            torch.cuda.synchronize()
        print("c2")
        # 正式测量
        total_tokens = 0
        t = time.time()
        for i in range(n_iters):
            print("c3")
            output_ids = worker.forward(input_ids)
            total_tokens += output_ids.size(-1) 
            print("c3") 
        print("c4")
        if device == 'cuda':
            torch.cuda.synchronize()
        # 计算时间和吞吐量
        total_time = time.time() - t
        total_samples = batch_size * n_iters
        speed = total_samples / total_time
        ms_per_token = (total_time * 1000) / total_tokens  # 计算每个token生成所需的毫秒数
        logger.info(f'Done, n_iters: {n_iters}, batch size: {batch_size}, image shape: {image_shape}, device type: {device.type}, kv mode: {worker.kv_mode}')
        logger.info(f'total time: {total_time} s, total samples: {total_samples}, throughput: {speed:.3f} samples/second.')
        logger.info(f'ms/token: {ms_per_token:.3f}')

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test different kv_modes for speed performance.")
    parser.add_argument('--warmup_iters', type=int, default=10, help='Number of warmup iterations.')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of test iterations.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--text_tokens', type=int, default=77, help='Number of text tokens.')
    parser.add_argument('--image_shape', type=tuple, default=(3, 224, 224), help='Shape of the input image.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for testing.')
    args = parser.parse_args()

    # 定义所有可能的kv_mode
    kv_modes = [
        "origin", "h2o", "weighted_merge", "pivot_merge",
        "text_prior_h2o", "text_prior_weighted_merge", "text_prior_pivot_merge",
        "snapkv", "avg_merge", "mean_h2o", "text_prior_avg_merge", "new"
    ]

    # 假设已经有一个定义好的模型
    tokenizer, model, processor, context_len = load_pretrained_model(
        model_path=args.model_dir,
        model_base=None,
        model_name=args.model_dir,
        device_map='cuda',
        kv_mode=args.kv_mode,
        hh_ratio=args.hh_ratio,
        recent_ratio=args.recent_ratio,
    )
    # worker = KvModeWorker(model=model, kv_mode="new")
    # speed_test(worker)  

    # 对每个kv_mode进行测试
    for kv_mode in kv_modes:
        worker = KvModeWorker(model=model, kv_mode=kv_mode)
        speed_test(worker, warmup_iters=args.warmup_iters, n_iters=args.n_iters, batch_size=args.batch_size, text_tokens=args.text_tokens, image_shape=args.image_shape, device=args.device)

if __name__ == "__main__":
    main()
