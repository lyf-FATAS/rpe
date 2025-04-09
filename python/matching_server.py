import argparse
from cv_bridge import CvBridge, CvBridgeError
import torch
from hloc import extractors, matchers
from hloc.utils.base_model import dynamic_load
import rospy
from rpe.srv import Matching, MatchingResponse
from std_msgs.msg import Int32MultiArray
import numpy as np
import cv2
import PIL.Image


feature_extraction_confs = {
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    "superpoint_max": {
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
            # "keypoint_threshold": 0.001,
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
            # 'detection_threshold': 40,
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


matching_confs = {
    "superpoint+lightglue": {
        "model": {
            "name": "lightglue",
            "features": "superpoint",
        },
    },
    "disk+lightglue": {
        "model": {
            "name": "lightglue",
            "features": "disk",
        },
    },
    "superglue": {
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 50,
        },
    },
    "superglue-fast": {
        "model": {
            "name": "superglue",
            "weights": "outdoor",
            "sinkhorn_iterations": 5,
        },
    },
    "NN-superpoint": {
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "distance_threshold": 0.7,
        },
    },
    "NN-ratio": {
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
            "ratio_threshold": 0.8,
        }
    },
    "NN-mutual": {
        "model": {
            "name": "nearest_neighbor",
            "do_mutual_check": True,
        },
    },
    "adalam": {
        "model": {"name": "adalam"},
    },
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
def detect_features(data):
    pred = extractor(
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
        uncertainty = getattr(extractor, "detection_noise", 1) * scales.mean()

    return pred


def form_pair(kps_and_desc0, kps_and_desc1, img_size0, img_size1):
    data = {}

    for k, v in kps_and_desc0.items():
        data[k + "0"] = (
            torch.from_numpy(v).unsqueeze(0).float().to(device, non_blocking=True)
        )
    data["image_size0"] = torch.from_numpy(img_size0).unsqueeze(0).float()
    # some matchers might expect an image but only use its size
    data["image0"] = torch.empty((1,) + tuple(img_size0)[::-1]).unsqueeze(0)

    for k, v in kps_and_desc1.items():
        data[k + "1"] = (
            torch.from_numpy(v).unsqueeze(0).float().to(device, non_blocking=True)
        )
    data["image_size1"] = torch.from_numpy(img_size1).unsqueeze(0).float()
    # some matchers might expect an image but only use its size
    data["image1"] = torch.empty((1,) + tuple(img_size1)[::-1]).unsqueeze(0)

    return data


@torch.no_grad()
def match_imgs(request):
    data1_l = preprocess_image(feature_extraction_conf["preprocessing"], request.img1_l)
    data1_r = preprocess_image(feature_extraction_conf["preprocessing"], request.img1_r)
    data2_l = preprocess_image(feature_extraction_conf["preprocessing"], request.img2_l)
    data2_r = preprocess_image(feature_extraction_conf["preprocessing"], request.img2_r)

    kps_and_desc1_l = detect_features(data1_l)
    kps_and_desc1_r = detect_features(data1_r)
    kps_and_desc2_l = detect_features(data2_l)
    kps_and_desc2_r = detect_features(data2_r)

    # Stereo matching for img1
    pair1_lr = form_pair(
        kps_and_desc1_l,
        kps_and_desc1_r,
        data1_l["original_size"],
        data1_r["original_size"],
    )
    match_stereo1 = matcher(pair1_lr)["matches0"][0].cpu().numpy()

    # Stereo matching for img2
    pair2_lr = form_pair(
        kps_and_desc2_l,
        kps_and_desc2_r,
        data2_l["original_size"],
        data2_r["original_size"],
    )
    match_stereo2 = matcher(pair2_lr)["matches0"][0].cpu().numpy()

    # Matching img1 and img2
    pair12_l = form_pair(
        kps_and_desc1_l,
        kps_and_desc2_l,
        data1_l["original_size"],
        data2_l["original_size"],
    )
    match12 = matcher(pair12_l)["matches0"][0].cpu().numpy()

    resp = MatchingResponse()
    resp.kps1_l = bridge.cv2_to_imgmsg(
        kps_and_desc1_l["keypoints"], encoding="passthrough"
    )
    resp.kps1_r = bridge.cv2_to_imgmsg(
        kps_and_desc1_r["keypoints"], encoding="passthrough"
    )
    resp.kps2_l = bridge.cv2_to_imgmsg(
        kps_and_desc2_l["keypoints"], encoding="passthrough"
    )
    resp.kps2_r = bridge.cv2_to_imgmsg(
        kps_and_desc2_r["keypoints"], encoding="passthrough"
    )
    resp.match_stereo1 = Int32MultiArray(data=match_stereo1.tolist())
    resp.match_stereo2 = Int32MultiArray(data=match_stereo2.tolist())
    resp.match12 = Int32MultiArray(data=match12.tolist())
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_extraction_conf",
        type=str,
        default="superpoint_max",
        choices=list(feature_extraction_confs.keys()),
    )
    parser.add_argument(
        "--matching_conf",
        type=str,
        default="superglue",
        choices=list(matching_confs.keys()),
    )
    args = parser.parse_args()
    feature_extraction_conf = feature_extraction_confs[args.feature_extraction_conf]
    matching_conf = matching_confs[args.matching_conf]

    bridge = CvBridge()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extraction_model = dynamic_load(
        extractors, feature_extraction_conf["model"]["name"]
    )
    extractor = (
        feature_extraction_model(feature_extraction_conf["model"]).eval().to(device)
    )
    matching_model = dynamic_load(matchers, matching_conf["model"]["name"])
    matcher = matching_model(matching_conf["model"]).eval().to(device)

    rospy.init_node("matching_server")
    s = rospy.Service("hloc_matching", Matching, match_imgs)
    print("============= Matching Server =============")
    rospy.spin()
