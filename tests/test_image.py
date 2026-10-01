"""Tests for image generation subsystem."""
import sys
import pytest
import asyncio
from unittest.mock import MagicMock, patch

# Mock diffusers before importing LocalStableDiffusionProvider
mock_diffusers = MagicMock()
sys.modules['diffusers'] = mock_diffusers

from robot.image import ImageJob, JobStatus, ImageGenerationService
from robot.image.sd_provider import LocalStableDiffusionProvider

@pytest.fixture
def memory_guard():
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 8 * 1024 * 1024 * 1024 # 8 GB
        yield mock_vm

def test_provider_initialization(memory_guard):
    provider = LocalStableDiffusionProvider(model_id="test-model")
    assert provider.model_id == "test-model"
    assert provider._backend == "cpu"
    assert provider._fallback is True

def test_missing_model_handling(memory_guard):
    provider = LocalStableDiffusionProvider(model_id="invalid-model")
    mock_diffusers.StableDiffusionPipeline.from_pretrained.side_effect = Exception("Model not found")
    with pytest.raises(Exception, match="Model not found"):
        provider._load_model()
    mock_diffusers.StableDiffusionPipeline.from_pretrained.side_effect = None

def test_memory_guard():
    with patch("psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 1 * 1024 * 1024 * 1024 # 1 GB
        provider = LocalStableDiffusionProvider()
        with pytest.raises(RuntimeError, match="Insufficient RAM"):
            provider._load_model()

@pytest.mark.asyncio
async def test_job_creation_and_queueing(memory_guard):
    svc = ImageGenerationService()
    job = svc.submit_job("test prompt", width=512, height=512, steps=4)
    assert job.status == JobStatus.QUEUED
    assert job.prompt == "test prompt"
    assert svc._queue.qsize() == 1

@pytest.mark.asyncio
async def test_job_cancellation(memory_guard):
    svc = ImageGenerationService()
    job = svc.submit_job("cancel me")
    
    res = await svc.cancel_job(job.job_id)
    assert res is True
    assert job.status == JobStatus.CANCELLED
    
@pytest.mark.asyncio
async def test_successful_generation(memory_guard):
    # Integration test using mock diffusers
    provider = LocalStableDiffusionProvider()
    provider._model_loaded = True
    provider._pipe = MagicMock()
    provider._pipe.return_value.images = [MagicMock()]
    
    job = ImageJob(prompt="test", steps=1)
    ev = asyncio.Event()
    
    with patch("psutil.Process") as mock_ps:
        mock_ps.return_value.memory_info.return_value.rss = 1000
        path = await provider.generate(job, ev)
        
    assert job.status == JobStatus.COMPLETED
    assert job.progress == 100
    assert path.endswith(".png")

@pytest.mark.asyncio
async def test_failed_generation(memory_guard):
    provider = LocalStableDiffusionProvider()
    provider._model_loaded = True
    provider._pipe = MagicMock(side_effect=RuntimeError("GPU crash"))
    
    job = ImageJob(prompt="test", steps=1)
    ev = asyncio.Event()
    
    with pytest.raises(RuntimeError, match="GPU crash"):
        await provider.generate(job, ev)
    
