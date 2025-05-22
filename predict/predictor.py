# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch
import numpy as np
from typing import List, Dict, Any

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.structures import ImageList, Instances

class BatchPredictor:
    """
    A predictor that supports batch processing of images.
    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        # Prepare transforms
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Args:
            images (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
        Returns:
            predictions (List[dict]): list of model outputs.
        """
        if not images:
            return []

        # Preprocess images
        batched_inputs = []
        for image in images:
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Expected numpy array, got {type(image)}")
            if len(image.shape) != 3:
                raise ValueError(f"Expected 3D array (H,W,C), got shape {image.shape}")

            if self.input_format == "RGB":
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            image = self.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            batched_inputs.append({
                "image": image,
                "height": height,
                "width": width
            })

        # Run inference
        with torch.no_grad():
            try:
                # Get model predictions
                outputs = self.model(batched_inputs)

                # Convert results to list of dicts
                predictions = []
                for i, output in enumerate(outputs):
                    height = batched_inputs[i]["height"]
                    width = batched_inputs[i]["width"]

                    # Convert to CPU and numpy
                    if hasattr(output, "to"):
                        output = output.to("cpu")
                    if hasattr(output, "numpy"):
                        output = output.numpy()

                    # Convert boxes to original image size if they exist
                    if "instances" in output:
                        instances = output["instances"]
                        if hasattr(instances, "pred_boxes"):
                            boxes = instances.pred_boxes.tensor
                            scale_x = width / batched_inputs[i]["image"].size(-1)
                            scale_y = height / batched_inputs[i]["image"].size(-2)
                            # Scale boxes manually
                            boxes[:, 0] *= scale_x  # x1
                            boxes[:, 1] *= scale_y  # y1
                            boxes[:, 2] *= scale_x  # x2
                            boxes[:, 3] *= scale_y  # y2

                    predictions.append(output)

                return predictions
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                raise

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False, confidence_threshold=0.5):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
            confidence_threshold (float): threshold for filtering predictions by confidence score
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.confidence_threshold = confidence_threshold

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(image)}")

        predictions = self.predictor([image])[0]  # Get first (and only) prediction
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        vis_output = None

        try:
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    # Filter instances by confidence threshold
                    instances = instances[instances.scores > self.confidence_threshold]
                    vis_output = visualizer.draw_instance_predictions(predictions=instances)
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            raise

        return predictions, vis_output

    def run_on_batch(self, images):
        """
        Args:
            images (List[np.ndarray]): a list of images of shape (H, W, C) (in BGR order).
        Returns:
            predictions (List[dict]): list of model outputs.
            vis_outputs (List[VisImage]): list of visualized image outputs.
        """
        if not images:
            return [], []

        # Get predictions for all images at once
        predictions = self.predictor(images)
        vis_outputs = []

        # Process each prediction and create visualizations
        for i, (pred, image) in enumerate(zip(predictions, images)):
            try:
                # Convert image from OpenCV BGR format to Matplotlib RGB format
                image = image[:, :, ::-1]
                visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
                vis_output = None

                if "panoptic_seg" in pred:
                    panoptic_seg, segments_info = pred["panoptic_seg"]
                    vis_output = visualizer.draw_panoptic_seg_predictions(
                        panoptic_seg.to(self.cpu_device), segments_info
                    )
                else:
                    if "sem_seg" in pred:
                        vis_output = visualizer.draw_sem_seg(
                            pred["sem_seg"].argmax(dim=0).to(self.cpu_device)
                        )
                    if "instances" in pred:
                        instances = pred["instances"].to(self.cpu_device)
                        # Filter instances by confidence threshold
                        instances = instances[instances.scores > self.confidence_threshold]
                        vis_output = visualizer.draw_instance_predictions(predictions=instances)

                vis_outputs.append(vis_output)
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                vis_outputs.append(None)

        return predictions, vis_outputs

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.
        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
