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

#初始化多种方法的配置参数
feature_extraction_confs = {
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


def resize_image(image, size, interp):  #传入参数：图像 目标尺寸 插值方法
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


def preprocess_image(prep_conf, img_msg): #图像预处理
    prep_conf = {**default_prep_conf, **prep_conf} #合并配置信息

    if prep_conf["grayscale"]:
        mode = "mono8"
    else:
        mode = "rgb8"

    try:
        image = bridge.imgmsg_to_cv2(img_msg, desired_encoding=mode) #将ROS图像消息转换为OpenCV格式的图像，使用指定的编码模式
    except CvBridgeError as e:
        print(e)

    if not prep_conf["grayscale"] and len(image.shape) == 3: #如果预设配置不是灰度图并且接收的图像是三通道的（即彩色图像），则将BGR格式的图像转换为RGB格式
        image = image[:, :, ::-1]  # BGR to RGB

    image = image.astype(np.float32) #将图像的数据类型转换为np.float32
    original_size = image.shape[:2][::-1] #获取原始图像的尺寸

    if prep_conf["resize_max"] and (
        prep_conf["resize_force"] or max(original_size) > prep_conf["resize_max"]
    ):
        scale = prep_conf["resize_max"] / max(original_size) #计算缩放比例
        size_new = tuple(int(round(x * scale)) for x in original_size)
        image = resize_image( #调整图像大小
            image, size_new, "cv2_area"
        )  # pil_linear is more accurate but slower

    if prep_conf["grayscale"]:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0 #归一化

    return { #返回一个字典
        "image": image,
        "original_size": np.array(original_size),
    }


@torch.no_grad()
def detect_features(data):
    pred = extractor( #将预处理后的图像数据传入模型model中进行推理，得到特征检测的结果pred
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

    return pred #返回一个字典


def form_pair(kps_and_desc0, kps_and_desc1, img_size0, img_size1):
    data = {}

    for k, v in kps_and_desc0.items(): #遍历kps_and_desc0中的关键点和描述符，将其转换为PyTorch张量并存储在data字典中，键名以"0"结尾表示属于第一个图像
        data[k + "0"] = (
            torch.from_numpy(v).unsqueeze(0).float().to(device, non_blocking=True)
        )
    data["image_size0"] = torch.from_numpy(img_size0).unsqueeze(0).float() #将第一个图像的尺寸信息存储在data字典中，并创建一个空的图像张量作为占位符，有些匹配器可能只需要使用图像尺寸而不需要实际图像
    # some matchers might expect an image but only use its size
    data["image0"] = torch.empty((1,) + tuple(img_size0)[::-1]).unsqueeze(0)

    for k, v in kps_and_desc1.items(): #遍历kps_and_desc1中的关键点和描述符，将其转换为PyTorch张量并存储在data字典中，键名以"1"结尾表示属于第二个图像
        data[k + "1"] = (
            torch.from_numpy(v).unsqueeze(0).float().to(device, non_blocking=True) #将第二个图像的尺寸信息存储在data字典中，并创建一个空的图像张量作为占位符，有些匹配器可能只需要使用图像尺寸而不需要实际图像
        )
    data["image_size1"] = torch.from_numpy(img_size1).unsqueeze(0).float()
    # some matchers might expect an image but only use its size
    data["image1"] = torch.empty((1,) + tuple(img_size1)[::-1]).unsqueeze(0)

    return data #返回一个字典


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
    match_stereo1 = matcher(pair1_lr)["matches0"][0].cpu().numpy() #对pair1_lr进行特征匹配，并获取匹配结果中的"matches0"字段的第一个元素，从GPU内存中转移到CPU上并转换为NumPy数组

    # Stereo matching for img2
    pair2_lr = form_pair(
        kps_and_desc2_l,
        kps_and_desc2_r,
        data2_l["original_size"],
        data2_r["original_size"],
    )
    match_stereo2 = matcher(pair2_lr)["matches0"][0].cpu().numpy() #对pair2_lr进行特征匹配，并获取匹配结果中的"matches0"字段的第一个元素，从GPU内存中转移到CPU上并转换为NumPy数组

    # Matching img1 and img2
    pair12_l = form_pair(
        kps_and_desc1_l,
        kps_and_desc2_l,
        data1_l["original_size"],
        data2_l["original_size"],
    )
    match12 = matcher(pair12_l)["matches0"][0].cpu().numpy() #对pair12_l进行特征匹配，并获取匹配结果中的"matches0"字段的第一个元素，从GPU内存中转移到CPU上并转换为NumPy数组

    resp = MatchingResponse()
    #将特征检测和特征匹配的结果转换位ROS图像消息回传给客户端
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
