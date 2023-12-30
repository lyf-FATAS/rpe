import argparse
from cv_bridge import CvBridge, CvBridgeError
import torch
from hloc import extractors
from hloc.utils.base_model import dynamic_load
import rospy
from rpe.srv import FeatureDetection, FeatureDetectionResponse
import numpy as np
import cv2
import PIL.Image


confs = {
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
            "resize_force": True,
        },
    },
    "r2d2": {
        "model": {
            "name": "r2d2",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1024,
        },
    },
    ############# Don't work *^* #############
    # "d2net-ss": {
    #     "model": {
    #         "name": "d2net",
    #         "multiscale": False,
    #     },
    #     "preprocessing": {
    #         "grayscale": False,
    #         "resize_max": 1600,
    #     },
    # },
    "sift": {
        "model": {"name": "dog"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "sosnet": {
        "model": {"name": "dog", "descriptor": "sosnet"},
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    },
    "disk": {
        "model": {
            "name": "disk",
            "max_keypoints": 5000,
        },
        "preprocessing": {
            "grayscale": False,
            "resize_max": 1600,
        },
    },
}


default_prep_conf = {
    "grayscale": False,
    "resize_max": None,
    "resize_force": False,
}


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


def preprocess_image(prep_conf, img_msg):
    prep_conf = {**default_prep_conf, **prep_conf}

    if prep_conf["grayscale"]:
        mode = "mono8"
    else:
        mode = "rgb8"

    try:
        image = bridge.imgmsg_to_cv2(img_msg, desired_encoding=mode)
    except CvBridgeError as e:
        print(e)

    if not prep_conf["grayscale"] and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB

    image = image.astype(np.float32)
    original_size = image.shape[:2][::-1]

    if prep_conf["resize_max"] and (
        prep_conf["resize_force"] or max(original_size) > prep_conf["resize_max"]
    ):
        scale = prep_conf["resize_max"] / max(original_size)
        size_new = tuple(int(round(x * scale)) for x in original_size)
        image = resize_image(
            image, size_new, "cv2_area"
        )  # pil_linear is more accurate but slower

    if prep_conf["grayscale"]:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0

    return {
        "image": image,
        "original_size": np.array(original_size),
    }


@torch.no_grad()
def detect_features(request):
    data = preprocess_image(conf["preprocessing"], request.img)

    pred = model(
        {
            "image": torch.from_numpy(data["image"])
            .unsqueeze(0)
            .to(device, non_blocking=True)
        }
    )
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

    # Postprocessing
    if "keypoints" in pred:
        size = np.array(data["image"].shape[-2:][::-1])
        scales = (data["original_size"] / size).astype(np.float32)
        pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
        if "scales" in pred:
            pred["scales"] *= scales.mean()
        # add keypoint uncertainties scaled to the original resolution
        uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

    resp = FeatureDetectionResponse()
    resp.kps = bridge.cv2_to_imgmsg(pred["keypoints"], encoding="passthrough")
    resp.desc = bridge.cv2_to_imgmsg(
        np.transpose(pred["descriptors"]), encoding="passthrough"
    )

    # print(pred["keypoints"])
    # print(np.transpose(pred["descriptors"]))

    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conf", type=str, default="superpoint_max", choices=list(confs.keys())
    )
    args = parser.parse_args()
    conf = confs[args.conf]

    bridge = CvBridge()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    rospy.init_node("feature_detection_server")
    s = rospy.Service("hloc_feature_detection", FeatureDetection, detect_features)
    print("============= Feature Detection Server, QiDong! =============")
    rospy.spin()
