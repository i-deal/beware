import wandb
from datetime import datetime
import subprocess

def initialize_wandb(project_name, config):
    """Initialize a WandB run with custom monitoring"""
    wandb.init(
        project=project_name,
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

def get_gpu_metrics():
    """Get GPU metrics using nvidia-smi"""
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', 
                '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total', 
                '--format=csv,noheader,nounits'
            ], encoding='utf-8'
        )
        
        gpu_stats = []
        for line in result.strip().split('\n'):
            temp, util, mem_used, mem_total = map(float, line.split(','))
            gpu_stats.append({
                'temperature': temp,
                'utilization': util,
                'memory_used': mem_used,
                'memory_total': mem_total
            })
        return gpu_stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None

def log_system_metrics():
    """Log system metrics including GPU usage, temperature, and memory"""
    try:
        # GPU metrics
        gpu_stats = get_gpu_metrics()
        if gpu_stats:
            for idx, stats in enumerate(gpu_stats):
                wandb.log({
                    f'gpu_{idx}/temperature': stats['temperature'],
                    f'gpu_{idx}/gpu_util_percent': stats['utilization'],
                    f'gpu_{idx}/memory_used_percent': (stats['memory_used'] / stats['memory_total']) * 100
                })
    except Exception as e:
        print(f"Unable to log GPU metrics: {e}")