# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops


class LocalizationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a localization model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import LocalizationPredictor

        args = dict(model='yolov8n.pt', source=ASSETS)
        predictor = LocalizationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the SegmentationPredictor with the provided configuration, overrides, and callbacks."""
        self.radii = overrides.pop("radii")
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "locate"

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression_loc(
            preds,
            self.args.conf,
            self.args.dor,
            radii=self.radii,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :2] = ops.scale_locations(img.shape[2:], pred[:, :2], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, locations=pred))
        return results
