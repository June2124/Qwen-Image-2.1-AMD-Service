"""Lazy ComfyUI child-process lifecycle and serialized inference."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, UnidentifiedImageError

from comfy_workflow import (
    DIFFUSION_MODEL,
    IMAGE_SIZE_MULTIPLE,
    MAX_CANVAS_SIDE,
    TEXT_ENCODER,
    VAE_MODEL,
    build_workflow,
    canvas_workflow_size,
)


logger = logging.getLogger(__name__)


class ComfyServiceError(RuntimeError):
    """ComfyUI could not complete a request."""


class ComfyServiceTimeout(ComfyServiceError):
    """ComfyUI exceeded a configured deadline."""


@dataclass(frozen=True)
class ManagerConfig:
    comfy_root: Path
    input_dir: Path
    output_dir: Path
    user_dir: Path
    frontend_root: Path
    log_path: Path
    host: str
    port: int
    startup_timeout: int
    inference_timeout: int
    idle_unload_seconds: int
    reserve_vram_gb: float
    cache_device: str
    cache_dtype: str
    reference_max_side: int

    @classmethod
    def from_env(cls) -> "ManagerConfig":
        data_root = Path(os.getenv("QWEN_DATA_ROOT", "/data"))
        return cls(
            comfy_root=Path(os.getenv("COMFY_ROOT", "/opt/ComfyUI")),
            input_dir=Path(os.getenv("COMFY_INPUT_DIR", str(data_root / "input"))),
            output_dir=Path(os.getenv("COMFY_OUTPUT_DIR", str(data_root / "output"))),
            user_dir=Path(os.getenv("COMFY_USER_DIR", str(data_root / "user"))),
            frontend_root=Path(os.getenv("COMFY_FRONTEND_ROOT", "/opt/comfy-empty-web")),
            log_path=Path(os.getenv("COMFY_LOG_PATH", str(data_root / "comfyui.log"))),
            host=os.getenv("COMFY_HOST", "127.0.0.1"),
            port=int(os.getenv("COMFY_PORT", "8188")),
            startup_timeout=int(os.getenv("COMFY_STARTUP_TIMEOUT", "300")),
            inference_timeout=int(os.getenv("QWEN_INFERENCE_TIMEOUT", "1800")),
            idle_unload_seconds=int(os.getenv("QWEN_IDLE_UNLOAD_SECONDS", "60")),
            reserve_vram_gb=float(os.getenv("COMFY_RESERVE_VRAM_GB", "1.0")),
            cache_device=os.getenv("QWEN_CACHE_DEVICE", "auto"),
            cache_dtype=os.getenv("QWEN_CACHE_DTYPE", "default"),
            reference_max_side=max(
                IMAGE_SIZE_MULTIPLE,
                min(
                    MAX_CANVAS_SIDE,
                    int(os.getenv("QWEN_REFERENCE_MAX_SIDE", "640")),
                ),
            ),
        )


class ComfyProcessManager:
    def __init__(self, config: Optional[ManagerConfig] = None):
        self.config = config or ManagerConfig.from_env()
        self._state_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reaper_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._log_handle = None
        self._busy = False
        self._queued = 0
        self._last_activity = time.time()
        self._last_error = ""
        for path in (
            self.config.input_dir,
            self.config.output_dir,
            self.config.user_dir,
            self.config.frontend_root,
            self.config.log_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def base_url(self) -> str:
        return "http://{}:{}".format(self.config.host, self.config.port)

    def start_reaper(self) -> None:
        with self._state_lock:
            if self._reaper_thread and self._reaper_thread.is_alive():
                return
            self._stop_event.clear()
            self._reaper_thread = threading.Thread(
                target=self._reaper_loop,
                name="qwen-comfy-idle-reaper",
                daemon=True,
            )
            self._reaper_thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self.stop(force=True, reason="service shutdown")

    def _reaper_loop(self) -> None:
        interval = max(5, min(30, self.config.idle_unload_seconds // 4 or 5))
        while not self._stop_event.wait(interval):
            with self._state_lock:
                idle_for = time.time() - self._last_activity
                should_stop = (
                    self.config.idle_unload_seconds > 0
                    and self._process_alive_locked()
                    and not self._busy
                    and self._queued == 0
                    and idle_for >= self.config.idle_unload_seconds
                )
            if should_stop:
                self.stop(force=False, reason="idle timeout")

    def _process_alive_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _command(self) -> List[str]:
        return [
            sys.executable,
            str(self.config.comfy_root / "main.py"),
            "--listen",
            self.config.host,
            "--port",
            str(self.config.port),
            "--disable-auto-launch",
            "--disable-metadata",
            "--disable-all-custom-nodes",
            "--disable-async-offload",
            "--disable-pinned-memory",
            "--preview-method",
            "none",
            "--input-directory",
            str(self.config.input_dir),
            "--output-directory",
            str(self.config.output_dir),
            "--user-directory",
            str(self.config.user_dir),
            "--front-end-root",
            str(self.config.frontend_root),
            "--reserve-vram",
            str(self.config.reserve_vram_gb),
        ]

    def ensure_started(self) -> None:
        with self._state_lock:
            if self._process_alive_locked():
                return
            if not (self.config.comfy_root / "main.py").is_file():
                raise ComfyServiceError("ComfyUI source is missing")
            self._close_log_locked()
            self._log_handle = self.config.log_path.open("a", encoding="utf-8")
            self._log_handle.write("\n=== starting ComfyUI {} ===\n".format(time.ctime()))
            self._log_handle.flush()
            self._process = subprocess.Popen(
                self._command(),
                cwd=str(self.config.comfy_root),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            process = self._process
            self._last_activity = time.time()
            logger.info("started ComfyUI child pid=%s", process.pid)

        deadline = time.monotonic() + self.config.startup_timeout
        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ComfyServiceError(
                    "ComfyUI exited during startup: {}".format(self._tail_log())
                )
            try:
                payload = self._request_json("GET", "/object_info/TextEncodeQwenImage21", timeout=5)
                if isinstance(payload, dict) and "TextEncodeQwenImage21" in payload:
                    return
                last_error = "Qwen Image 2.1 nodes are unavailable"
            except Exception as exc:  # startup retries are expected
                last_error = str(exc)
            time.sleep(2)
        self.stop(force=True, reason="startup timeout")
        raise ComfyServiceTimeout(
            "ComfyUI did not become ready: {}; {}".format(last_error, self._tail_log())
        )

    def stop(self, *, force: bool, reason: str) -> bool:
        with self._state_lock:
            if self._busy and not force:
                return False
            process = self._process
            self._process = None
        if process is None:
            with self._state_lock:
                self._close_log_locked()
            return False
        logger.info("stopping ComfyUI child pid=%s reason=%s", process.pid, reason)
        try:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait(timeout=10)
        except ProcessLookupError:
            pass
        finally:
            with self._state_lock:
                self._last_activity = time.time()
                self._close_log_locked()
        return True

    def _close_log_locked(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            finally:
                self._log_handle = None

    @contextlib.contextmanager
    def _inference_slot(self):
        with self._state_lock:
            self._queued += 1
        self._inference_lock.acquire()
        with self._state_lock:
            self._queued -= 1
            self._busy = True
            self._last_activity = time.time()
        try:
            yield
        finally:
            with self._state_lock:
                self._busy = False
                self._last_activity = time.time()
            self._inference_lock.release()

    def generate(
        self,
        *,
        mode: str,
        prompt: str,
        negative_prompt: str,
        size: str,
        steps: int,
        guidance_scale: float,
        seed: int,
        image_blobs: Iterable[bytes],
        close_model: bool,
        output_size: Optional[tuple[int, int]] = None,
    ) -> bytes:
        with self._inference_slot():
            request_id = uuid.uuid4().hex
            input_paths: List[Path] = []
            output_path: Optional[Path] = None
            try:
                self.ensure_started()
                image_names, input_paths = self._stage_images(
                    request_id, image_blobs, mode=mode
                )
                workflow = build_workflow(
                    mode=mode,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    size=size,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    image_names=image_names,
                    filename_prefix="qwen-image-2.1/{}".format(request_id),
                    cache_device=self.config.cache_device,
                    cache_dtype=self.config.cache_dtype,
                )
                prompt_id = self._submit(workflow)
                output_info = self._wait_for_output(prompt_id)
                raw = self._fetch_output(output_info)
                raw = self._crop_output(raw, output_size)
                output_path = self._output_path(output_info)
                with self._state_lock:
                    self._last_error = ""
                    self._last_activity = time.time()
                if close_model:
                    self.stop(force=True, reason="close_model requested")
                return raw
            except Exception as exc:
                with self._state_lock:
                    self._last_error = str(exc)[:1000]
                self.stop(force=True, reason="inference failure")
                raise
            finally:
                for path in input_paths:
                    with contextlib.suppress(OSError):
                        path.unlink()
                if output_path is not None:
                    with contextlib.suppress(OSError):
                        output_path.unlink()
                    with contextlib.suppress(OSError):
                        output_path.parent.rmdir()

    @staticmethod
    def _edge_pad(image: Image.Image) -> Image.Image:
        width, height = image.size
        padded_width, padded_height = canvas_workflow_size(width, height)
        if (padded_width, padded_height) == (width, height):
            return image

        padded = Image.new(image.mode, (padded_width, padded_height))
        padded.paste(image, (0, 0))
        if padded_width > width:
            right_edge = image.crop((width - 1, 0, width, height)).resize(
                (padded_width - width, height)
            )
            padded.paste(right_edge, (width, 0))
        if padded_height > height:
            bottom_edge = padded.crop((0, height - 1, padded_width, height)).resize(
                (padded_width, padded_height - height)
            )
            padded.paste(bottom_edge, (0, height))
        return padded

    def _prepare_input_image(
        self, source: Image.Image, *, is_canvas: bool
    ) -> Image.Image:
        mode = "RGBA" if "A" in source.getbands() else "RGB"
        image = source.convert(mode)
        if not is_canvas:
            longest_side = max(image.size)
            if longest_side > self.config.reference_max_side:
                scale = self.config.reference_max_side / float(longest_side)
                width = max(1, int(image.width * scale))
                height = max(1, int(image.height * scale))
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image = image.resize((width, height), resampling)
        return self._edge_pad(image)

    def _stage_images(
        self, request_id: str, image_blobs: Iterable[bytes], *, mode: str
    ):
        names = []
        paths = []
        for index, blob in enumerate(image_blobs, start=1):
            name = "qwen-{}-{:02d}.png".format(request_id, index)
            path = self.config.input_dir / name
            try:
                with Image.open(io.BytesIO(blob)) as source:
                    source.load()
                    if source.width * source.height > 40_000_000:
                        raise ComfyServiceError("input image is too large")
                    prepared = self._prepare_input_image(
                        source,
                        is_canvas=(mode == "image-to-image" and index == 1),
                    )
                    prepared.save(path, format="PNG", optimize=False)
            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as exc:
                raise ComfyServiceError("input image is invalid") from exc
            names.append(name)
            paths.append(path)
        return names, paths

    @staticmethod
    def _crop_output(raw: bytes, output_size: Optional[tuple[int, int]]) -> bytes:
        if output_size is None:
            return raw
        width, height = output_size
        try:
            with Image.open(io.BytesIO(raw)) as source:
                source.load()
                if source.size == (width, height):
                    return raw
                if source.width < width or source.height < height:
                    raise ComfyServiceError(
                        "generated image is smaller than the editing canvas"
                    )
                cropped = source.crop((0, 0, width, height))
                output = io.BytesIO()
                cropped.save(output, format="PNG", optimize=False)
                return output.getvalue()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ComfyServiceError("generated image could not be cropped") from exc

    def _submit(self, workflow: dict) -> str:
        payload = self._request_json(
            "POST",
            "/prompt",
            payload={"prompt": workflow, "client_id": uuid.uuid4().hex},
            timeout=30,
        )
        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not prompt_id:
            raise ComfyServiceError("ComfyUI rejected the workflow: {}".format(payload))
        return str(prompt_id)

    def _wait_for_output(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self.config.inference_timeout
        while time.monotonic() < deadline:
            payload = self._request_json(
                "GET", "/history/{}".format(urllib.parse.quote(prompt_id)), timeout=15
            )
            history = payload.get(prompt_id) if isinstance(payload, dict) else None
            if history:
                status = history.get("status") or {}
                if status.get("status_str") == "error":
                    raise ComfyServiceError(
                        "ComfyUI execution failed: {}".format(status.get("messages"))
                    )
                outputs = history.get("outputs") or {}
                for node_output in outputs.values():
                    images = node_output.get("images") if isinstance(node_output, dict) else None
                    if images:
                        return images[0]
                if status.get("completed"):
                    raise ComfyServiceError("ComfyUI completed without an image")
            time.sleep(2)
        raise ComfyServiceTimeout("Qwen image inference timed out")

    def _fetch_output(self, output_info: dict) -> bytes:
        query = urllib.parse.urlencode(
            {
                "filename": output_info.get("filename", ""),
                "subfolder": output_info.get("subfolder", ""),
                "type": output_info.get("type", "output"),
            }
        )
        request = urllib.request.Request(self.base_url + "/view?" + query, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except Exception as exc:
            raise ComfyServiceError("could not read the generated image") from exc
        if not raw:
            raise ComfyServiceError("ComfyUI returned an empty image")
        return raw

    def _output_path(self, output_info: dict) -> Optional[Path]:
        filename = os.path.basename(str(output_info.get("filename") or ""))
        subfolder = str(output_info.get("subfolder") or "").replace("\\", "/").strip("/")
        if not filename or ".." in subfolder.split("/"):
            return None
        candidate = (self.config.output_dir / subfolder / filename).resolve()
        try:
            candidate.relative_to(self.config.output_dir.resolve())
        except ValueError:
            return None
        return candidate

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Optional[dict] = None,
        timeout: int,
    ) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise ComfyServiceError(
                "ComfyUI HTTP {}: {}".format(exc.code, detail or exc.reason)
            ) from exc
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            raise ComfyServiceError("ComfyUI request failed: {}".format(exc)) from exc

    def _tail_log(self, lines: int = 30) -> str:
        try:
            values = self.config.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return "no ComfyUI log available"
        return " | ".join(values[-lines:])[-4000:]

    def status(self) -> dict:
        with self._state_lock:
            process_alive = self._process_alive_locked()
            pid = self._process.pid if process_alive else None
            idle_for = max(0.0, time.time() - self._last_activity)
            return {
                "comfy_running": process_alive,
                "comfy_pid": pid,
                "busy": self._busy,
                "queued": self._queued,
                "idle_for_seconds": round(idle_for, 1),
                "idle_unload_seconds": self.config.idle_unload_seconds,
                "cache_device": self.config.cache_device,
                "cache_dtype": self.config.cache_dtype,
                "reference_max_side": self.config.reference_max_side,
                "last_error": self._last_error,
                "models": {
                    DIFFUSION_MODEL: (
                        self.config.comfy_root / "models" / "diffusion_models" / DIFFUSION_MODEL
                    ).is_file(),
                    TEXT_ENCODER: (
                        self.config.comfy_root / "models" / "text_encoders" / TEXT_ENCODER
                    ).is_file(),
                    VAE_MODEL: (
                        self.config.comfy_root / "models" / "vae" / VAE_MODEL
                    ).is_file(),
                },
            }


__all__ = [
    "ManagerConfig",
    "ComfyProcessManager",
    "ComfyServiceError",
    "ComfyServiceTimeout",
]
