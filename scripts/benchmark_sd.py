import time
import os
import psutil
from robot.image.sd_provider import LocalStableDiffusionProvider
from robot.image import ImageJob
import asyncio

async def run_benchmark():
    provider = LocalStableDiffusionProvider()
    print("Stable Diffusion Benchmark\n")
    print(f"Device: {provider._device}")
    print(f"Backend: {provider._backend}")
    print("Resolution: 512x512\n")
    
    # Load model first to separate load time
    t0 = time.time()
    provider._load_model()
    t1 = time.time()
    print(f"Model load time: {t1-t0:.2f} s\n")
    
    steps_to_test = [4, 8, 12, 20]
    print("Steps\tGeneration")
    
    for steps in steps_to_test:
        job = ImageJob(prompt="A realistic humanoid robot standing in a modern laboratory, cinematic lighting, highly detailed", steps=steps)
        ev = asyncio.Event()
        
        start = time.time()
        await provider.generate(job, ev)
        end = time.time()
        
        print(f"{steps}\t{end-start:.2f} s")
        
    peak_ram = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"\nPeak RAM: {peak_ram:.0f} MB")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
