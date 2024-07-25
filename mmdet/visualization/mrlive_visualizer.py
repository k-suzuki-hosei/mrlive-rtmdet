# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import cv2
import mmcv
import numpy as np
import time ####

try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer

from mmengine.visualization.utils import tensor2ndarray ####

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS
from ..structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from .palette import _get_adaptive_scales, get_palette, jitter_color

@VISUALIZERS.register_module()
class DetLocalVisualizer(Visualizer):
    """MMDetection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        bbox_color (str, tuple(int), optional): Color of bbox lines.
            The tuple of color should be in BGR order. Defaults to None.
        text_color (str, tuple(int), optional): Color of texts.
            The tuple of color should be in BGR order.
            Defaults to (200, 200, 200).
        mask_color (str, tuple(int), optional): Color of masks.
            The tuple of color should be in BGR order.
            Defaults to None.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmdet.structures import DetDataSample
        >>> from mmdet.visualization import DetLocalVisualizer

        >>> det_local_visualizer = DetLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_instances = InstanceData()
        >>> gt_instances.bboxes = torch.Tensor([[1, 2, 2, 5]])
        >>> gt_instances.labels = torch.randint(0, 2, (1,))
        >>> gt_det_data_sample = DetDataSample()
        >>> gt_det_data_sample.gt_instances = gt_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample)
        >>> det_local_visualizer.add_datasample(
        ...                       'image', image, gt_det_data_sample,
        ...                        out_file='out_file.jpg')
        >>> det_local_visualizer.add_datasample(
        ...                        'image', image, gt_det_data_sample,
        ...                         show=True)
        >>> pred_instances = InstanceData()
        >>> pred_instances.bboxes = torch.Tensor([[2, 4, 4, 8]])
        >>> pred_instances.labels = torch.randint(0, 2, (1,))
        >>> pred_det_data_sample = DetDataSample()
        >>> pred_det_data_sample.pred_instances = pred_instances
        >>> det_local_visualizer.add_datasample('image', image,
        ...                         gt_det_data_sample,
        ...                         pred_det_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 bbox_color: Optional[Union[str, Tuple[int]]] = None,
                 text_color: Optional[Union[str,
                                            Tuple[int]]] = (200, 200, 200),
                 mask_color: Optional[Union[str, Tuple[int]]] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.bbox_color = bbox_color
        self.text_color = text_color
        self.mask_color = mask_color
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `DetLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray, instances: ['InstanceData'],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]]) -> np.ndarray:  ##
        print("................_draw_instance......................")
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        if 'masks' in instances:
            time_mask = time.time()
            labels = instances.labels
            masks = instances.masks

            """
            3~4ms → 5~9ms
            マスク画像のデータをnumpy形式へ変換
            マスク画像の値をbool値変換
            """
            time_tensor = time.time() ##
            # 3~5ms
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu()  # https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html
                masks = masks.numpy()
            elif isinstance(masks, (PolygonMasks, BitmapMasks)):
                masks = masks.to_ndarray()
            masks = masks.astype(bool)

            print("time_tensor={:.4f}ms" .format((time.time()-time_tensor)*1000)) ##


            #-----------------------------------------------------------------------------------------------------------------------------------------------------------#
            """
            200~220ms → 26~33ms
            マスク情報をもとに結果画像にマスクを描画
            """
            time_polygons = time.time() ##

            if masks.ndim == 2:
                masks = masks[None]
            assert image.shape[:2] == masks.shape[
                            1:], '`binary_marks` must have ' \
                                    'the same shape with image'

            black_image = np.zeros_like(image)
            color = (255, 255, 255)

            for i in range(len(masks[:, 0, 0])):
                if labels[i].item() in  [0,2]:  # ここでlabelから映したい物体を選択する。62がモニターかな？
                    black_image[:,:][masks[i]] = color

            print("time_binary={:.4f}ms" .format((time.time()-time_polygons)*1000)) ##
            print("draw_instance={:.4f}" .format(1000*(time.time()-time_mask)))
            print("................_draw_instance......................")
            return black_image

        
    @master_only   ####rtmの動作ではここが使われていた。
    def mask_add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:  ##
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """

        """
        2=3ms
        結果画像のピクセル値をunit8へ変換
        classesをデータセットから取得
        paletteをデータセットから取得
        """
        print("/////////add_datasample///////////")
        time_prepare = time.time()  ###
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = None  ####使わない
        gt_img_data = None
        pred_img_data = None
        print("prepare={:.4f}ms".format(1000*(time.time()-time_prepare)))

        """
        300~380ms → 41~60ms
        結果画像にマスク画像を描画（draw_instances）
        """
        time_draw_pred = time.time()  ####
        if draw_pred and data_sample is not None:
            print("||||||||||||||||||||start draw_instances||||||||||||||||||||||")
            if 'pred_instances' in data_sample:
                t1 = time.time()
                pred_instances = data_sample.pred_instances
                print("t1={:.4f}ms" .format((time.time()-t1)*1000))

                # ここのpred_instanceに時間がかかる
                t2 = time.time()
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr]
                print("t2={:.4f}ms" .format((time.time()-t2)*1000))
                print(pred_instances.device)

                t3 = time.time()
                pred_img_data = self._draw_instances(image, pred_instances,
                                                    classes, palette)
                print("t3={:.4f}ms" .format((time.time()-t3)*1000))

            if 'pred_sem_seg' in data_sample:
                pred_img_data = self._draw_sem_seg(pred_img_data,
                                                data_sample.pred_sem_seg,
                                                classes, palette)

            if 'pred_panoptic_seg' in data_sample:
                assert classes is not None, 'class information is ' \
                                            'not provided when ' \
                                            'visualizing panoptic ' \
                                            'segmentation results.'
                pred_img_data = self._draw_panoptic_seg(
                    pred_img_data, data_sample.pred_panoptic_seg.numpy(),
                    classes, palette)
            print("||||||||||||||||||||end draw_instances||||||||||||||||||||||")
        print("add_draw_instance={:.4f}ms" .format((time.time()-time_draw_pred)*1000))  ####

        """
        1~2ms
        """
        time_else = time.time() ####------------------------------------------------------------------------------------1.0ms--
        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None: ##ここ
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        cutout_img = cv2.bitwise_and(image, drawn_img)  #### cutout 1ms

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:  ##ここ 0ms
            self.add_image(name, drawn_img, step)
        print("add_time_else={:.4f}ms" .format((time.time()-time_else)*1000)) ####----------------------------------
        print("/////////add_datasample///////////")
        return cutout_img, drawn_img ####

def random_color(seed):
    """Random a color according to the input seed."""
    if sns is None:
        raise RuntimeError('motmetrics is not installed,\
                 please install it by: pip install seaborn')
    np.random.seed(seed)
    colors = sns.color_palette()
    color = colors[np.random.choice(range(len(colors)))]
    color = tuple([int(255 * c) for c in color])
    return color


@VISUALIZERS.register_module()
class TrackLocalVisualizer(Visualizer):
    """Tracking Local Visualizer for the MOT, VIS tasks.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
                Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(name, image, vis_backends, save_dir)
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `TrackLocalVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_instances(self, image: np.ndarray,
                        instances: InstanceData) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        classes = self.dataset_meta.get('classes', None)

        # get colors and texts
        # for the MOT and VIS tasks
        colors = [random_color(_id) for _id in instances.instances_id]
        categories = [
            classes[label] if classes is not None else f'cls{label}'
            for label in instances.labels
        ]
        if 'scores' in instances:
            texts = [
                f'{category_name}\n{instance_id} | {score:.2f}'
                for category_name, instance_id, score in zip(
                    categories, instances.instances_id, instances.scores)
            ]
        else:
            texts = [
                f'{category_name}\n{instance_id}' for category_name,
                instance_id in zip(categories, instances.instances_id)
            ]

        # draw bboxes and texts
        if 'bboxes' in instances:
            # draw bboxes
            bboxes = instances.bboxes.clone()
            self.draw_bboxes(
                bboxes,
                edge_colors=colors,
                alpha=self.alpha,
                line_widths=self.line_width)
            # draw texts
            if texts is not None:
                positions = bboxes[:, :2] + self.line_width
                areas = (bboxes[:, 3] - bboxes[:, 1]) * (
                    bboxes[:, 2] - bboxes[:, 0])
                scales = _get_adaptive_scales(areas.cpu().numpy())
                for i, pos in enumerate(positions):
                    self.draw_texts(
                        texts[i],
                        pos,
                        colors='black',
                        font_sizes=int(13 * scales[i]),
                        bboxes=[{
                            'facecolor': [c / 255 for c in colors[i]],
                            'alpha': 0.8,
                            'pad': 0.7,
                            'edgecolor': 'none'
                        }])

        # draw masks
        if 'masks' in instances:
            masks = instances.masks
            polygons = []
            for i, mask in enumerate(masks):
                contours, _ = bitmap_to_polygon(mask)
                polygons.extend(contours)
            self.draw_polygons(polygons, edge_colors='w', alpha=self.alpha)
            self.draw_binary_masks(masks, colors=colors, alphas=self.alpha)

        return self.get_image()

    @master_only  
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: DetDataSample = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: int = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.
        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (OptTrackSampleList): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT TrackDataSample.
                Default to True.
            draw_pred (bool): Whether to draw Prediction TrackDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (int): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            assert 'gt_instances' in data_sample
            gt_img_data = self._draw_instances(image, data_sample.gt_instances)

        if draw_pred and data_sample is not None:
            assert 'pred_track_instances' in data_sample
            pred_instances = data_sample.pred_track_instances
            if 'scores' in pred_instances:
                pred_instances = pred_instances[
                    pred_instances.scores > pred_score_thr].cpu()
            pred_img_data = self._draw_instances(image, pred_instances)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)
