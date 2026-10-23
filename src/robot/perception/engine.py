"""Continuous lightweight perception pipeline using OpenCV with decoupled 10 FPS background worker.

Features:
- Decoupled background worker thread with bounded frame queue (size=2) dropping stale frames
- Fast YuNet INT8 ONNX face detection (people tracking, face position, attention estimate)
- Real-time YOLOv8 object detection with 3D tabletop spatial mapping and temporal smoothing
  (inference cadence configurable via ``yolo_fps`` / ROBOT_YOLO_FPS, default 3 Hz so CPU
  YOLO never starves realtime audio; HUD and WorldState reuse cached detections between ticks)
- Adaptive frame differencing for physical motion / activity detection
- Scene ambient illumination / lighting classification (dark, dim, normal, bright)
- Steady 10 FPS WorldState publishing with CPU throttling (never starves audio playback or STT)
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from robot.perception.camera import CameraCapture, silence_c_stderr
from robot.perception.world import DetectedObject, DetectedPerson, TabletopItem, WorldState
from robot.telemetry import get_logger

logger = get_logger("perception.engine")


class PerceptionEngine:
    """Continuous visual perception engine that updates WorldState at 10 FPS."""

    def __init__(
        self,
        camera: Optional[CameraCapture] = None,
        yunet_model_path: str = "models/vision/face_detection_yunet.onnx",
        yolo_model_path: str = "models/vision/yolov8n.pt",
        target_fps: float = 10.0,
        enable_motion: bool = True,
        yolo_fps: Optional[float] = None,
    ) -> None:
        """Initialize the perception engine with camera and detectors.

        Args:
            camera: Optional CameraCapture instance.
            yunet_model_path: Path to YuNet ONNX face detection model.
            yolo_model_path: Path to YOLOv8 object detection weights.
            target_fps: Target inference frequency in FPS (default: 10.0).
            enable_motion: Whether to calculate motion levels via frame differencing.
            yolo_fps: Independent YOLOv8 cadence in Hz. Defaults to the
                ROBOT_YOLO_FPS environment value, else 3.0 Hz — running
                YOLOv8n at the full 10 Hz target_fps on CPU starves realtime
                audio. A frame-skip counter gates inference; everything else
                (annotation, WorldState) keeps running at target_fps using the
                cached detections between YOLO ticks.
        """
        self.camera = camera or CameraCapture()
        self.yunet_model_path = yunet_model_path
        self.yolo_model_path = yolo_model_path
        self.target_fps = target_fps
        self.enable_motion = enable_motion

        if yolo_fps is None:
            env_val = os.environ.get("ROBOT_YOLO_FPS", "")
            try:
                yolo_fps = float(env_val) if env_val else 3.0
            except ValueError:
                yolo_fps = 3.0
        self.yolo_fps = max(0.5, float(yolo_fps))

        # Internal frame-skip counter: run YOLO roughly every target_fps/yolo_fps frames.
        self._yolo_frame_skip = max(1, int(round(self.target_fps / self.yolo_fps)))
        self._frame_counter = 0
        self._last_yolo_objects: list[DetectedObject] = []

        self.world_state = WorldState()
        self._running = False
        self._feeder_task: Optional[asyncio.Task] = None
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._yolo: Any = None
        self._prev_gray: Optional[np.ndarray] = None
        self._last_yunet_size: tuple[int, int] = (0, 0)
        self._annotated_lock = threading.Lock()
        self._latest_annotated_frame: Optional[np.ndarray] = None

        # Decoupled bounded frame queue (size=2) to guarantee zero latency buildup
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop_event = threading.Event()

        # AMD Radeon 680M iGPU VLM grounding engine
        from robot.perception.vlm import LocalVLMGrounding

        self.vlm = LocalVLMGrounding(enabled=True, device="amd_igpu")

    def _init_yolo(self) -> bool:
        """Initialize lightweight YOLOv8 object detector."""
        model_p = Path(self.yolo_model_path)
        if not model_p.exists():
            logger.warning("yolo_model_not_found", path=self.yolo_model_path)
            return False

        try:
            from ultralytics import YOLO

            self._yolo = YOLO(str(model_p))
            logger.info("yolo_detector_initialized", model=self.yolo_model_path)
            return True
        except Exception as e:
            logger.error("yolo_init_failed", error=str(e))
            self._yolo = None
            return False

    @property
    def is_paused(self) -> bool:
        """Return True if visual perception is temporarily paused."""
        return self.camera.is_paused

    def pause(self) -> None:
        """Pause visual perception and release camera hardware."""
        self.camera.pause()
        with self._annotated_lock:
            self._latest_annotated_frame = None
        self.world_state.lighting = "offline"
        self.world_state.motion = "none"
        self.world_state.people = []
        self.world_state.objects = []
        self.world_state.updated_at = time.time()
        logger.info("perception_engine_paused")

    def resume(self) -> None:
        """Resume visual perception and reacquire camera hardware."""
        self.camera.resume()
        logger.info("perception_engine_resumed")

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return the latest frame drawn with green polygons and perception overlays."""
        if self.camera.is_paused:
            return None
        with self._annotated_lock:
            if self._latest_annotated_frame is not None:
                return self._latest_annotated_frame.copy()
        return self.camera.get_latest_frame()

    def _init_yunet(self) -> bool:
        """Initialize OpenCV YuNet Face Detector if model file exists."""
        model_p = Path(self.yunet_model_path)
        if not model_p.exists():
            logger.warning("yunet_model_not_found", path=self.yunet_model_path)
            return False

        try:
            with silence_c_stderr():
                self._detector = cv2.FaceDetectorYN_create(
                    model=str(model_p),
                    config="",
                    input_size=(320, 240),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=10,
                )
            self._last_yunet_size = (320, 240)
            logger.info("yunet_detector_initialized", model=self.yunet_model_path)
            return True
        except Exception as e:
            logger.error("yunet_init_failed", error=str(e))
            self._detector = None
            return False

    async def start(self) -> None:
        """Start camera capture and background 10 FPS perception worker thread."""
        if self._running:
            return

        self.camera.start()
        self._init_yunet()
        self._init_yolo()
        self._running = True
        self._worker_stop_event.clear()

        # Start dedicated background worker thread
        self._worker_thread = threading.Thread(
            target=self._perception_worker_loop,
            name="PerceptionWorkerThread",
            daemon=True,
        )
        self._worker_thread.start()

        # Start camera feeder task in asyncio event loop
        self._feeder_task = asyncio.create_task(
            self._frame_feeder_loop(), name="PerceptionFeederLoop"
        )
        logger.info("perception_engine_started", target_fps=self.target_fps)

    async def stop(self) -> None:
        """Stop perception analysis and release resources."""
        self._running = False
        self._worker_stop_event.set()

        if self._feeder_task:
            self._feeder_task.cancel()
            try:
                await self._feeder_task
            except asyncio.CancelledError:
                pass
            self._feeder_task = None

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

        # Drain frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except Exception:
                break

        self.camera.stop()
        logger.info("perception_engine_stopped")

    async def _frame_feeder_loop(self) -> None:
        """Feed latest frames from camera into bounded worker queue (maxsize=2)."""
        interval = 1.0 / max(1.0, self.target_fps * 1.5)

        while self._running:
            try:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    # Drop stale frame if queue is full to prevent backlog delay
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except Exception:
                            pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
            except Exception as e:
                logger.error("perception_feeder_error", error=str(e))

            await asyncio.sleep(interval)

    def _perception_worker_loop(self) -> None:
        """Dedicated background worker running visual inference at steady 10 FPS."""
        target_interval = 1.0 / max(0.5, self.target_fps)

        while not self._worker_stop_event.is_set():
            t_start = time.time()
            try:
                # Block briefly for fresh frame
                frame = self._frame_queue.get(timeout=0.2)
                self._analyze_frame(frame)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("perception_worker_exception", error=str(e))

            elapsed = time.time() - t_start
            sleep_time = max(0.005, target_interval - elapsed)
            time.sleep(sleep_time)

    def _analyze_frame(self, frame: np.ndarray) -> None:
        """Process frame for faces, motion, ambient light, and tabletop items."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Motion analysis via frame differencing
        motion_level = "none"
        motion_score = 0.0
        if self.enable_motion:
            small_gray = cv2.resize(gray, (160, 120))
            if self._prev_gray is not None:
                diff = cv2.absdiff(small_gray, self._prev_gray)
                mean_diff = float(np.mean(diff))
                motion_score = mean_diff
                if mean_diff > 25.0:
                    motion_level = "high"
                elif mean_diff > 10.0:
                    motion_level = "medium"
                elif mean_diff > 3.0:
                    motion_level = "low"
            self._prev_gray = small_gray

        # 2. Lighting conditions
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 30:
            lighting = "dark"
        elif mean_brightness < 70:
            lighting = "dim"
        elif mean_brightness > 200:
            lighting = "bright"
        else:
            lighting = "normal"

        # 3. People / Face detection via YuNet
        people: list[DetectedPerson] = []
        if self._detector is not None:
            scale_w = 320.0 / w
            scale_h = 240.0 / h
            input_img = cv2.resize(frame, (320, 240))

            if self._last_yunet_size != (320, 240):
                self._detector.setInputSize((320, 240))
                self._last_yunet_size = (320, 240)

            _, faces = self._detector.detect(input_img)
            if faces is not None:
                for idx, face in enumerate(faces):
                    conf = float(face[-1])
                    if conf < 0.6:
                        continue

                    x = int(face[0] / scale_w)
                    y = int(face[1] / scale_h)
                    box_w = int(face[2] / scale_w)
                    box_h = int(face[3] / scale_h)

                    center_x = x + box_w / 2.0
                    center_y = y + box_h / 2.0
                    norm_x = max(0.0, min(1.0, center_x / max(1, w)))
                    norm_y = max(0.0, min(1.0, center_y / max(1, h)))

                    if center_x < w * 0.35:
                        position = "left"
                    elif center_x > w * 0.65:
                        position = "right"
                    else:
                        position = "center"

                    face_area_ratio = (box_w * box_h) / (w * h)
                    if face_area_ratio > 0.08:
                        distance = "near"
                    elif face_area_ratio > 0.02:
                        distance = "medium"
                    else:
                        distance = "far"

                    people.append(
                        DetectedPerson(
                            id=f"user_{idx + 1}" if idx > 0 else "user",
                            position=position,
                            distance_estimate=distance,
                            confidence=round(conf, 2),
                            looking_at_robot=True,
                            bbox=[x, y, box_w, box_h],
                            norm_x=round(norm_x, 3),
                            norm_y=round(norm_y, 3),
                        )
                    )

        # 4. Object Detection & Tabletop Analysis via YOLOv8 (cadence-gated)
        objects: list[DetectedObject] = []
        tabletop_items: list[TabletopItem] = []
        run_yolo = False
        if self._yolo is not None:
            skip = max(1, int(round(self.target_fps / self.yolo_fps)))
            run_yolo = (self._frame_counter % skip) == 0
        self._frame_counter += 1

        if run_yolo:
            try:
                results = self._yolo.predict(
                    frame,
                    verbose=False,
                    conf=0.22,
                    imgsz=320,
                    device="cpu",
                )
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    names = results[0].names
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        label = names.get(cls_id, f"obj_{cls_id}")
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()
                        ox1, oy1, ox2, oy2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        obw = max(1, ox2 - ox1)
                        obh = max(1, oy2 - oy1)

                        ocx = ox1 + obw / 2.0
                        ocy = oy1 + obh / 2.0
                        norm_cx = ocx / max(1, w)
                        norm_cy = ocy / max(1, h)

                        if ocx < w * 0.35:
                            opos = "left"
                        elif ocx > w * 0.65:
                            opos = "right"
                        else:
                            opos = "center"

                        objects.append(
                            DetectedObject(
                                label=label,
                                confidence=round(conf, 2),
                                bbox=[ox1, oy1, obw, obh],
                                position=opos,
                            )
                        )

                        # Classify tabletop items (lower half or desk area)
                        if norm_cy > 0.35 and label.lower() not in ("person",):
                            # Map clock position
                            if norm_cx < 0.25:
                                clock_pos = "9 o'clock"
                            elif norm_cx < 0.40:
                                clock_pos = "11 o'clock"
                            elif norm_cx <= 0.60:
                                clock_pos = "12 o'clock"
                            elif norm_cx <= 0.75:
                                clock_pos = "1 o'clock"
                            else:
                                clock_pos = "3 o'clock"

                            # Distance estimation based on size ratio
                            area_ratio = (obw * obh) / (w * h)
                            if area_ratio > 0.12:
                                dist_str = "near"
                                dist_m = 0.35
                            elif area_ratio > 0.04:
                                dist_str = "medium"
                                dist_m = 0.70
                            else:
                                dist_str = "far"
                                dist_m = 1.20

                            tabletop_items.append(
                                TabletopItem(
                                    object_name=label,
                                    confidence=round(conf, 2),
                                    bounding_box=[ox1, oy1, obw, obh],
                                    estimated_distance=dist_str,
                                    estimated_distance_m=dist_m,
                                    position_clock=clock_pos,
                                    last_seen=time.time(),
                                )
                            )
            except Exception as e:
                logger.debug("yolo_predict_error", error=str(e))
            self._last_yolo_objects = objects
        elif self._yolo is not None:
            # Cadence-skipped frame: reuse the last YOLO detections so HUD
            # boxes never flicker between YOLO ticks. Tabletop temporal
            # tracking is only refreshed on YOLO ticks; the 3s expiration
            # window comfortably covers the ~1/yolo_fps gap.
            objects = list(self._last_yolo_objects)

        # 5. Generate high-tech HUD annotated visual frame.
        # `frame` is a private per-analysis copy pulled from the bounded queue
        # (CameraCapture.get_latest_frame already returns .copy()); nothing
        # retains or mutates it elsewhere in this function before this point
        # (cvtColor/resize allocate new buffers, YOLO predict does not mutate
        # its input and results.orig_img is never used), so we draw directly
        # onto it and skip the extra 640x480 copy.
        annotated = frame

        # Render Detected Objects (omit person if already tracked by YuNet)
        for obj in objects:
            if obj.label.lower() == "person" and len(people) > 0:
                continue
            ox, oy, obw, obh = obj.bbox
            opts = np.array([
                [ox, oy],
                [ox + obw, oy],
                [ox + obw, oy + obh],
                [ox, oy + obh]
            ], np.int32).reshape((-1, 1, 2))

            cv2.polylines(annotated, [opts], isClosed=True, color=(0, 230, 180), thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(annotated, (ox + obw // 2, oy + obh // 2), 3, (0, 230, 180), -1)

            obj_tag = f"{obj.label.upper()} {int(obj.confidence * 100)}%"
            (otw, oth), _ = cv2.getTextSize(obj_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            tag_oy = max(oth + 4, oy - 4)
            cv2.rectangle(annotated, (ox, tag_oy - oth - 3), (ox + otw + 4, tag_oy + 2), (4, 18, 14), -1)
            cv2.rectangle(annotated, (ox, tag_oy - oth - 3), (ox + otw + 4, tag_oy + 2), (0, 230, 180), 1)
            cv2.putText(annotated, obj_tag, (ox + 2, tag_oy - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 200), 1, cv2.LINE_AA)

        # Render Detected People via Sleek Cyber Reticle & Corner Brackets
        for p in people:
            x, y, bw, bh = p.bbox
            corner_len = max(8, min(18, bw // 4, bh // 4))

            # Top-left corner
            cv2.line(annotated, (x, y), (x + corner_len, y), (0, 255, 128), 2, cv2.LINE_AA)
            cv2.line(annotated, (x, y), (x, y + corner_len), (0, 255, 128), 2, cv2.LINE_AA)
            # Top-right corner
            cv2.line(annotated, (x + bw, y), (x + bw - corner_len, y), (0, 255, 128), 2, cv2.LINE_AA)
            cv2.line(annotated, (x + bw, y), (x + bw, y + corner_len), (0, 255, 128), 2, cv2.LINE_AA)
            # Bottom-left corner
            cv2.line(annotated, (x, y + bh), (x + corner_len, y + bh), (0, 255, 128), 2, cv2.LINE_AA)
            cv2.line(annotated, (x, y + bh), (x, y + bh - corner_len), (0, 255, 128), 2, cv2.LINE_AA)
            # Bottom-right corner
            cv2.line(annotated, (x + bw, y + bh), (x + bw - corner_len, y + bh), (0, 255, 128), 2, cv2.LINE_AA)
            cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - corner_len), (0, 255, 128), 2, cv2.LINE_AA)

            # Center target crosshair dot
            cx, cy = x + bw // 2, y + bh // 2
            cv2.circle(annotated, (cx, cy), 3, (0, 255, 128), -1)

            # Sleek compact badge (avoids blocking face or running off frame)
            tag_text = f"USER • {int(p.confidence * 100)}%"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tag_y = max(th + 5, y - 4)
            cv2.rectangle(annotated, (x, tag_y - th - 4), (x + tw + 6, tag_y + 2), (8, 20, 14), -1)
            cv2.rectangle(annotated, (x, tag_y - th - 4), (x + tw + 6, tag_y + 2), (0, 255, 128), 1)
            cv2.putText(annotated, tag_text, (x + 3, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 128), 1, cv2.LINE_AA)

        with self._annotated_lock:
            self._latest_annotated_frame = annotated

        # Update WorldState with temporal tabletop tracking
        self.world_state.people = people
        self.world_state.objects = objects
        self.world_state.update_tabletop_tracking(tabletop_items)
        self.world_state.motion = motion_level
        self.world_state.motion_score = round(motion_score, 2)
        self.world_state.lighting = lighting
        self.world_state.scene = "desk/indoor"
        self.world_state.updated_at = time.time()

    def inspect_tabletop(self) -> dict[str, Any]:
        """Return structured tabletop spatial inspection result.

        Returns:
            Dictionary with items_count, items list, scene_lighting, and motion.
        """
        items = [item.to_dict() for item in self.world_state.tabletop_items]
        return {
            "status": "success",
            "items_count": len(items),
            "items": items,
            "scene_lighting": self.world_state.lighting,
            "motion": self.world_state.motion,
            "timestamp": time.time(),
        }

    def get_latest_scene(self) -> dict[str, Any]:
        """Provide a structured snapshot of the most recently perceived scene.

        Aggregates detected faces, people count, YOLO visual objects, tabletop items,
        scene lighting, and motion state from current WorldState.

        Returns:
            dict[str, Any]: Scene snapshot dictionary containing num_faces,
                people, objects, tabletop_items, motion, lighting, and scene.
        """
        objects = [
            {
                "label": obj.label,
                "confidence": round(obj.confidence, 2),
                "position": obj.position,
            }
            for obj in self.world_state.objects
        ]
        people = [
            {
                "id": p.id,
                "confidence": round(p.confidence, 2),
                "position": p.position,
                "distance": p.distance_estimate,
            }
            for p in self.world_state.people
        ]
        return {
            "num_faces": len(self.world_state.people),
            "people": people,
            "objects": objects,
            "tabletop_items": [item.to_dict() for item in self.world_state.tabletop_items],
            "motion": self.world_state.motion,
            "lighting": self.world_state.lighting,
            "scene": self.world_state.scene,
            "timestamp": self.world_state.updated_at,
        }

