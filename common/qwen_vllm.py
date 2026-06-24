"""Shared vLLM engine for batched Qwen2.5-VL classify + caption (s5/s8)."""

from __future__ import annotations

import gc
import logging
import os
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

from common.gpu_info import log_service_gpus, resolve_gpu_ids
from common.caption_models import resolve_caption_model
from common.paths import qwen_classify_model_path, qwen_video_model_path

logger = logging.getLogger(__name__)

ImageInput = Union[Image.Image, Sequence[Image.Image]]


def configure_vllm_env() -> None:
    """Tune vLLM env for installed version (0.8.x vs 0.19+)."""
    try:
        import vllm as _vllm

        major_minor = tuple(int(x) for x in _vllm.__version__.split(".")[:2])
    except Exception:
        major_minor = (0, 8)
    if major_minor < (0, 19):
        os.environ.setdefault("VLLM_USE_V1", "0")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    # TP worker subprocesses must rendezvous on loopback (node IP often unreachable).
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


configure_vllm_env()


def _fresh_master_port() -> None:
    """Pick a free port so a prior vLLM run cannot block TP rendezvous."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(sock.getsockname()[1])


_ORIGINAL_CUDA_VISIBLE_DEVICES: Optional[str] = None


def _remember_cuda_visible_devices() -> None:
    """Save launcher GPU mask before Ray steps can clobber it."""
    global _ORIGINAL_CUDA_VISIBLE_DEVICES
    if _ORIGINAL_CUDA_VISIBLE_DEVICES is None:
        _ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


def _restore_cuda_visible_devices() -> None:
    """Ray driver processes may rewrite CUDA_VISIBLE_DEVICES (e.g. to '0')."""
    if _ORIGINAL_CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _ORIGINAL_CUDA_VISIBLE_DEVICES


def _restore_distributed_env_for_vllm() -> None:
    """Ray/torch.distributed can leave a stale node IP as MASTER_ADDR."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"
    _fresh_master_port()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_NAME",
        "LOCAL_WORLD_SIZE",
    ):
        os.environ.pop(key, None)


def _cleanup_vllm_runtime() -> None:
    import time
    import torch

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _restore_distributed_env_for_vllm()
    time.sleep(3)


def shutdown_ray_after_service() -> None:
    """Tear down pipeline Ray and restore GPU visibility for vLLM TP."""
    _remember_cuda_visible_devices()
    try:
        import ray

        if ray.is_initialized():
            logger.info("Shutting down Ray after service")
            ray.shutdown()
    except ImportError:
        pass
    _restore_cuda_visible_devices()
    _restore_distributed_env_for_vllm()


def _shutdown_ray_before_vllm() -> None:
    """Ray (s6/s7) and vLLM multiprocessing conflict if Ray stays initialized."""
    shutdown_ray_after_service()


def _visible_device_list() -> List[str]:
    """Physical GPU ids from the launcher's CUDA_VISIBLE_DEVICES."""
    _remember_cuda_visible_devices()
    cvd = _ORIGINAL_CUDA_VISIBLE_DEVICES or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return []
    return [p.strip() for p in cvd.split(",") if p.strip() != ""]


def _physical_visible_for_logical(logical_gpu: int) -> str:
    """Map logical cuda:N to the physical GPU id for CUDA_VISIBLE_DEVICES masking."""
    parts = _visible_device_list()
    if 0 <= logical_gpu < len(parts):
        return parts[logical_gpu]
    return str(logical_gpu)


class _CudaVisibleScope:
    """Temporarily pin the process to a single physical GPU (data-parallel replica)."""

    def __init__(self, visible: str):
        self._new = visible
        self._old: Optional[str] = None

    def __enter__(self) -> "_CudaVisibleScope":
        self._old = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = self._new
        return self

    def __exit__(self, *_args: object) -> None:
        if self._old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = self._old


def _split_gpu_groups(gpu_ids: List[int], tensor_parallel_size: int) -> List[List[int]]:
    """Split GPUs into TP groups. TP=1 → one GPU per group (data parallel)."""
    if not gpu_ids:
        return [[]]
    tp = max(1, tensor_parallel_size)
    groups = [gpu_ids[i : i + tp] for i in range(0, len(gpu_ids), tp)]
    return [g for g in groups if g]


class _VLLMReplica:
    """Single vLLM tensor-parallel group on a disjoint GPU set."""

    def __init__(
        self,
        *,
        model_path: str,
        gpu_ids: List[int],
        max_model_len: int,
        max_images: int,
        max_tokens: int,
        replica_id: int,
        caption_family: str = "qwen",
        input_modality: str = "image",
        num_frames: int = 32,
        max_videos: int = 1,
    ):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.max_model_len = max_model_len
        self.max_images = max_images
        self.max_tokens = max_tokens
        self.replica_id = replica_id
        self.caption_family = caption_family
        self.input_modality = input_modality
        self.num_frames = num_frames
        self.max_videos = max_videos
        self._llm = None
        self._processor = None

    def load(self) -> None:
        if self._llm is not None:
            return
        import torch
        from transformers import AutoProcessor
        from vllm import LLM

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable inside vLLM worker. "
                "PyTorch driver mismatch — run: bash scripts/install_vllm.sh"
            )

        _restore_distributed_env_for_vllm()
        _fresh_master_port()
        tp = max(1, len(self.gpu_ids))
        executor = "mp" if tp > 1 else "uni"
        load_ctx = (
            _CudaVisibleScope(_physical_visible_for_logical(self.gpu_ids[0]))
            if tp == 1 and len(self.gpu_ids) == 1
            else nullcontext()
        )
        if self.input_modality == "video":
            mm_limit = {"video": self.max_videos}
        else:
            mm_limit = {"image": self.max_images}
        with load_ctx:
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._llm = LLM(
                model=self.model_path,
                tensor_parallel_size=tp,
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=self.max_model_len,
                limit_mm_per_prompt=mm_limit,
                enforce_eager=True,
                distributed_executor_backend=executor,
                gpu_memory_utilization=float(
                    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
                ),
            )
        logger.info(
            "vLLM replica %d ready on GPUs %s",
            self.replica_id,
            self.gpu_ids,
        )

    def _build_prompt(
        self,
        images: ImageInput,
        text: str,
        *,
        system_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(images, Image.Image):
            img_list = [images]
        else:
            img_list = list(images)

        content: List[Dict[str, Any]] = []
        for img in img_list:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})

        if system_text or self.caption_family == "gemma":
            sys_msg = system_text or ""
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
                {"role": "user", "content": content},
            ]
        else:
            messages = [{"role": "user", "content": content}]
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        mm_data = {"image": img_list[0]} if len(img_list) == 1 else {"image": img_list}
        return {"prompt": prompt_text, "multi_modal_data": mm_data}

    def _build_video_prompt(
        self,
        clip_path: str,
        text: str,
        *,
        system_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Native-video prompt: feed a clip MP4 as (frames, metadata) to vLLM."""
        from vllm.multimodal.video import OpenCVVideoBackend

        with open(clip_path, "rb") as fh:
            frames, metadata = OpenCVVideoBackend.load_bytes(
                fh.read(), num_frames=self.num_frames
            )
        content: List[Dict[str, Any]] = [
            {"type": "video"},
            {"type": "text", "text": text},
        ]
        if system_text:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": content},
            ]
        else:
            messages = [{"role": "user", "content": content}]
        prompt_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt_text,
            "multi_modal_data": {"video": (frames, metadata)},
        }

    def _unpack_item(
        self, item: Union[Tuple[ImageInput, str], Tuple[ImageInput, str, str]]
    ) -> Tuple[ImageInput, str, Optional[str]]:
        if len(item) >= 3:
            return item[0], item[1], item[2]
        return item[0], item[1], None

    def generate_batch(
        self,
        items: List[Union[Tuple[ImageInput, str], Tuple[ImageInput, str, str]]],
        *,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        if not items:
            return []

        from vllm import SamplingParams

        self.load()
        sampling = SamplingParams(
            temperature=0,
            max_tokens=max_tokens or self.max_tokens,
        )
        prompts = []
        for item in items:
            media, text, system_text = self._unpack_item(item)
            if self.input_modality == "video":
                prompts.append(
                    self._build_video_prompt(media, text, system_text=system_text)
                )
            else:
                prompts.append(
                    self._build_prompt(media, text, system_text=system_text)
                )
        outputs = self._llm.generate(prompts, sampling)
        return [out.outputs[0].text.strip() for out in outputs]

    def generate_chunks(
        self,
        items: List[Union[Tuple[ImageInput, str], Tuple[ImageInput, str, str]]],
        *,
        batch_size: int,
        max_tokens: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ) -> List[str]:
        from common.progress import progress_batched

        results: List[str] = []
        desc = progress_desc or "vLLM infer"
        for chunk in progress_batched(items, batch_size, desc=desc):
            results.extend(self.generate_batch(chunk, max_tokens=max_tokens))
        return results

    def generate_video_candidates(
        self,
        clip_path: str,
        text: str,
        *,
        n: int,
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """N sampled caption candidates for one clip (used to re-roll a clip
        whose first caption over-tagged a second person)."""
        from vllm import SamplingParams

        self.load()
        prompt = self._build_video_prompt(clip_path, text)
        sampling = SamplingParams(
            n=max(1, n),
            temperature=max(0.01, temperature),
            max_tokens=max_tokens or self.max_tokens,
        )
        outputs = self._llm.generate([prompt], sampling)
        return [o.text.strip() for o in outputs[0].outputs]

    def cleanup(self) -> None:
        llm = self._llm
        self._llm = None
        self._processor = None
        if llm is not None:
            try:
                engine = getattr(llm, "llm_engine", None)
                if engine is not None and hasattr(engine, "shutdown"):
                    engine.shutdown()
            except Exception as exc:
                logger.warning("vLLM replica %d shutdown: %s", self.replica_id, exc)
            del llm
        _cleanup_vllm_runtime()


class QwenVLLMEngine:
    """vLLM batched inference — one or more TP replicas (not s5 and s8 simultaneously)."""

    _shared: Optional["QwenVLLMEngine"] = None

    def __init__(self, config: Dict[str, Any], stage: str = "vlm"):
        self._config = config
        self.stage = stage
        vllm_cfg = config.get("models", {}).get("vllm", {})
        pcfg = config.get("pipeline", {})
        s5 = pcfg.get("s5", {})
        cap = pcfg.get("captioner", {})
        mp = pcfg.get("master_pipeline", {})

        # Multimodal input defaults (overridden to video for gemma4_dense vLLM-video).
        self.input_modality = "image"
        self.num_frames = int(cap.get("num_frames", vllm_cfg.get("num_frames", 32)))
        self.max_videos = int(vllm_cfg.get("max_videos_per_prompt", 1))

        if stage == "s5":
            self.model_path = str(
                s5.get("classify_model_path") or qwen_classify_model_path(config)
            )
            self._caption_family = "qwen"
            self.max_tokens = int(s5.get("max_tokens", 128))
            gpu_ids = [int(g) for g in s5.get("gpu_ids", mp.get("classify_gpu_ids", [0]))]
            tp = int(s5.get("tensor_parallel_size", 1))
            self.batch_size = int(s5.get("batch_size", vllm_cfg.get("batch_size", 16)))
        elif stage == "s4":
            s4 = pcfg.get("s4", {})
            if s4.get("model_path"):
                self.model_path = str(s4["model_path"])
            else:
                resolved = resolve_caption_model(config)
                if resolved["key"] == "gemma4":
                    self.model_path = str(resolved["model_path"])
                else:
                    root = Path(config.get("pipeline", {}).get("models_root", ""))
                    if not root:
                        from common.paths import models_root

                        root = models_root(config)
                    self.model_path = str(root / "gemma-4-31b-it")
            self._caption_family = "gemma"
            self.max_tokens = int(s4.get("max_tokens", 128))
            gpu_ids = [int(g) for g in s4.get("gpu_ids", [0, 1])]
            tp = int(s4.get("tensor_parallel_size", 1))
            self.batch_size = int(s4.get("batch_size", 8))
        else:
            resolved = resolve_caption_model(config)
            self.model_path = str(resolved["model_path"])
            self._caption_family = resolved["family"]
            self.input_modality = (
                "video" if resolved.get("backend") == "vllm_video" else "image"
            )
            self.max_tokens = int(
                vllm_cfg.get("max_tokens", cap.get("max_tokens", 1000))
            )
            gpu_ids = [int(g) for g in vllm_cfg.get("gpu_ids", cap.get("gpu_ids", [0]))]
            tp = int(cap.get("tensor_parallel_size", vllm_cfg.get("tensor_parallel_size", 0)))
            self.batch_size = int(
                cap.get("batch_size", vllm_cfg.get("batch_size", 16))
            )

        self.gpu_ids = resolve_gpu_ids(gpu_ids)
        if not self.gpu_ids:
            raise RuntimeError(
                "No CUDA GPUs visible for vLLM. "
                "Check CUDA_VISIBLE_DEVICES and PyTorch CUDA build "
                "(run: bash scripts/install_vllm.sh to fix cu130/cu124 mismatch)."
            )
        self.max_model_len = int(vllm_cfg.get("max_model_len", 8192))
        self.max_images = int(vllm_cfg.get("max_images_per_prompt", 3))
        if tp <= 0:
            tp = len(self.gpu_ids)
        self.tensor_parallel_size = tp
        self.gpu_groups = _split_gpu_groups(self.gpu_ids, tp)
        self._replicas: List[_VLLMReplica] = []

    @classmethod
    def acquire(cls, config: Dict[str, Any], stage: str = "vlm") -> "QwenVLLMEngine":
        if cls._shared is not None and cls._shared.stage != stage:
            cls.release()
        if cls._shared is None:
            cls._shared = cls(config, stage=stage)
        return cls._shared

    @classmethod
    def release(cls) -> None:
        if cls._shared:
            cls._shared.cleanup()
            cls._shared = None

    def _vllm_available(self) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def load(self) -> None:
        if self._replicas:
            return
        _shutdown_ray_before_vllm()
        _restore_distributed_env_for_vllm()
        if not self._vllm_available():
            raise ImportError(
                "vLLM is not installed. Run: bash scripts/install_vllm.sh"
            )
        if not Path(self.model_path).joinpath("config.json").exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        n_rep = len(self.gpu_groups)
        tp_w = len(self.gpu_groups[0])
        if n_rep == 1:
            tp_note = f"TP={tp_w}"
        elif tp_w == 1:
            tp_note = f"data-parallel ×{n_rep} (TP=1 per GPU)"
        else:
            tp_note = f"{n_rep}×TP={tp_w} tensor-parallel"
        log_service_gpus(
            self.stage,
            f"vLLM batched inference — {tp_note}",
            self.model_path,
            self.gpu_ids,
        )

        for idx, group in enumerate(self.gpu_groups):
            replica = _VLLMReplica(
                model_path=self.model_path,
                gpu_ids=group,
                max_model_len=self.max_model_len,
                max_images=self.max_images,
                max_tokens=self.max_tokens,
                replica_id=idx,
                caption_family=getattr(self, "_caption_family", "qwen"),
                input_modality=self.input_modality,
                num_frames=self.num_frames,
                max_videos=self.max_videos,
            )
            replica.load()
            self._replicas.append(replica)
        _restore_cuda_visible_devices()

    def generate_batch(
        self,
        items: List[Tuple[ImageInput, str]],
        *,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """Batched generate for (image[s], prompt) pairs."""
        if not items:
            return []
        return self.generate_chunks(items, batch_size=len(items), max_tokens=max_tokens)

    def _generate_parallel(
        self,
        items: List[Tuple[ImageInput, str]],
        *,
        batch_size: int,
        max_tokens: Optional[int],
    ) -> List[str]:
        n = len(self._replicas)
        chunk_size = (len(items) + n - 1) // n
        parts = [items[i * chunk_size : (i + 1) * chunk_size] for i in range(n)]
        ordered: List[Optional[List[str]]] = [None] * n

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {
                pool.submit(
                    self._replicas[i].generate_chunks,
                    parts[i],
                    batch_size=batch_size,
                    max_tokens=max_tokens,
                ): i
                for i in range(n)
                if parts[i]
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                ordered[idx] = fut.result()

        out: List[str] = []
        for part in ordered:
            if part:
                out.extend(part)
        return out

    def generate_chunks(
        self,
        items: List[Tuple[ImageInput, str]],
        *,
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ) -> List[str]:
        if not items:
            return []

        self.load()
        bs = batch_size or self.batch_size
        desc = progress_desc or f"{self.stage} vLLM"
        if len(self._replicas) == 1:
            return self._replicas[0].generate_chunks(
                items,
                batch_size=bs,
                max_tokens=max_tokens,
                progress_desc=desc,
            )
        return self._generate_parallel(items, batch_size=bs, max_tokens=max_tokens)

    def recaption_video_candidates(
        self,
        clip_path: str,
        text: str,
        *,
        n: int = 4,
        temperature: float = 0.5,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        """N sampled caption candidates for one clip (re-roll an over-tagged clip)."""
        self.load()
        if not self._replicas:
            return []
        return self._replicas[0].generate_video_candidates(
            clip_path, text, n=n, temperature=temperature, max_tokens=max_tokens
        )

    def cleanup(self) -> None:
        for replica in self._replicas:
            replica.cleanup()
        self._replicas = []
