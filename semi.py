import json
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
from datasets import transforms as T
from util import misc
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import OrderedDict

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098]]
CLASSES = ['0']


class EMA:
    def __init__(self, model):
        self.model = model
        self.decay = 0.9998
        self.gamma = 4
        self.interval = 3
        self.iter = 0
        self.shadow = {}
        self.backup = {}

    def register(self, shadow=None):
        if shadow is None:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.data.clone()
        else:
            self.shadow = shadow
            self.iter = 1000
            for name, param in self.model.named_parameters():
                if param.requires_grad and name not in self.shadow:
                    self.shadow[name] = param.data.clone()

    def update(self):
        self.iter += 1
        if self.iter % self.interval == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    momentum = min(self.decay, 1 - self.gamma / (self.gamma + self.iter))
                    new_average = (1.0 - momentum) * param.data + momentum * self.shadow[name].to(param.device)
                    self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].to(param.device)
        self.backup = {}


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b


def creat_empty_index(path):
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_folder_path = path + '/semi'
    image_files = os.listdir(image_folder_path)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder_path, image_file)
        image = Image.open(image_path)
        width, height = image.size

        image_info = {
            "id": idx + 1,
            "width": width,
            "height": height,
            "file_name": image_file,
        }
        coco_annotations["images"].append(image_info)
    with open(path + "/annotations/empty_coco_for_semi.json", "w") as f:
        json.dump(coco_annotations, f)


def solarize(img):
    img = np.array(img)
    ratio = random.random()
    threshold = 255 * ratio
    img = np.where(img < threshold, img, 255 - img)
    return Image.fromarray(img)


def histogramEqualization(img):
    img = np.array(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR))


def randomColor(image):
    transform_type = random.choice(['solarize', 'Brightness', 'Contrast', 'Sharpness', 'Color', 'autocontrast', 'histogramEqualization', 'posterize'])
    if transform_type == 'Color':
        random_factor = np.random.randint(10, 190) / 100.
        image = ImageEnhance.Color(image).enhance(random_factor)
    if transform_type == 'Brightness':
        random_factor = np.random.randint(10, 190) / 100.
        image = ImageEnhance.Brightness(image).enhance(random_factor)
    if transform_type == 'Contrast':
        random_factor = np.random.randint(10, 190) / 100.
        image = ImageEnhance.Contrast(image).enhance(random_factor)
    if transform_type == 'Sharpness':
        random_factor = np.random.randint(10, 190) / 100.
        image = ImageEnhance.Sharpness(image).enhance(random_factor)
    if transform_type == 'autocontrast':
        image = ImageOps.autocontrast(image)
    if transform_type == 'posterize':
        random_factor = random.randint(4, 8)
        image = ImageOps.posterize(image,random_factor)
    if transform_type == 'solarize':
        image = solarize(image)
    if transform_type == 'histogramEqualization':
        image = histogramEqualization(image)
    return image


def plot_results_noprob(pil, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil)
    ax = plt.gca()
    colors = COLORS * 100
    for (xmin, ymin, xmax, ymax), c in zip(boxes['boxes'], colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
    plt.axis('off')
    plt.savefig("xx.png")
    plt.show()


def rotate_rectangle(target, angle, h, w):
    theta = np.radians(angle)

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    x1, y1, x2, y2 = target
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    rotated_points = points - np.array([0, 0])
    fix = rotated_points.dot(rotation_matrix)

    target[0] = max(min(fix[:, 0]), 0)
    target[1] = max(min(fix[:, 1]), 0)
    target[2] = min(max(fix[:, 0]), w)
    target[3] = min(max(fix[:, 1]), h)
    if target[0] >= w or target[1] >= h or target[2] <= 0 or target[3] <= 0 or target[0] >= target[2] or target[1] >= \
            target[3]:
        return target, 0
    return target, 1


def deletebox(target, delidx):
    newboxes = []
    newlabels = []
    newscores = []
    newiscrowd = []
    for i in range(len(delidx)):
        if delidx[i] == 1:
            newboxes.append(target['boxes'][i])
            newlabels.append(target['labels'][i])
            newscores.append(target['scores'][i])
            newiscrowd.append(target['iscrowd'][i])
    if len(newboxes) > 0:
        target['boxes'] = torch.stack(newboxes, dim=0)
        target['labels'] = torch.stack(newlabels, dim=0)
        target['scores'] = torch.stack(newscores, dim=0)
        target['iscrowd'] = torch.stack(newiscrowd, dim=0)
    else:
        target['boxes'] = torch.empty((0, 4))
        target['labels'] = torch.empty((0,), dtype=torch.int64)
        target['scores'] = torch.empty((0,))
        target['iscrowd'] = torch.empty((0,))
    return target


def rotate(image, target, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    delidx = []
    for i in range(len(target['boxes'])):
        target['boxes'][i], idx = rotate_rectangle(target['boxes'][i], angle, h, w)
        delidx.append(idx)
    target = deletebox(target, delidx)
    return cv2.warpAffine(image, M, (w, h)), target


def translate_rectangle(target, xmove, ymove, h, w):
    target[0] = max(0, target[0] + xmove)
    target[1] = max(0, target[1] + ymove)
    target[2] = min(w, target[2] + xmove)
    target[3] = min(h, target[3] + ymove)
    if target[0] >= w or target[1] >= h or target[2] <= 0 or target[3] <= 0 or target[0] >= target[2] or target[1] >= \
            target[3]:
        return target, 0
    return target, 1


def translate(img_array, target, x_ratio, y_ratio):
    h, w = img_array.shape[:2]
    translate_x = int(w * x_ratio)
    translate_y = int(h * y_ratio)
    translation_matrix = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    translated_image = cv2.warpAffine(img_array, translation_matrix, (w, h))
    delidx = []
    for i in range(len(target['boxes'])):
        target['boxes'][i], idx = translate_rectangle(target['boxes'][i], translate_x, translate_y, h, w)
        delidx.append(idx)
    target = deletebox(target, delidx)
    return translated_image, target


def shear_rectangle(target, alpha, h, w):
    target[0] += min(w * alpha * (target[1] / (h - 1)), w * alpha * (target[3] / (h - 1)))
    target[2] += max(w * alpha * (target[1] / (h - 1)), w * alpha * (target[3] / (h - 1)))
    target[0] = max(0, target[0])
    target[2] = min(w, target[2])
    if target[0] >= w or target[2] <= 0 or target[0] >= target[2]:
        return target, 0
    return target, 1


def shear(image, target, alpha):
    h, w = image.shape[:2]
    shifted_image = np.zeros_like(image)

    for i in range(h):
        scale = (i / (h - 1)) * alpha * w
        new_x = int(np.clip(scale, 1 - w, w - 1))
        if new_x > 0:
            shifted_image[i, new_x:] = image[i, :-new_x]
        elif new_x < 0:
            shifted_image[i, :new_x] = image[i, -new_x:]
        else:
            shifted_image[i, :] = image[i, :]
    delidx = []
    for i in range(len(target['boxes'])):
        target['boxes'][i], idx = shear_rectangle(target['boxes'][i], alpha, h, w)
        delidx.append(idx)
    target = deletebox(target, delidx)
    return shifted_image, target


def rand_geometric(img, target):
    img = np.array(img)[:, :, :3]
    transform_type = random.choice(['rotate', 'translate', 'shear'])
    if transform_type == 'rotate':
        angle = random.random() * 60 - 30
        img, target = rotate(img, target, angle)
    elif transform_type == 'translate':
        ratiox = random.random() * 0.2 - 0.1
        ratioy = random.random() * 0.2 - 0.1
        img, target = translate(img, target, ratiox, ratioy)
    elif transform_type == 'shear':
        ratio = random.random() * 0.6 - 0.3
        img, target = shear(img, target, ratio)
    return Image.fromarray(np.uint8(img)), target


def cutout(image, target, patches=(1, 20), ratio=(0, 0.1), thr=0.7):
    image = np.array(image)[:, :, :3]
    height, width, _ = image.shape
    num_patches = np.random.randint(patches[0], patches[1] + 1)
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_patches):
        patch_width = int(width * np.random.uniform(ratio[0], ratio[1]))
        patch_height = int(height * np.random.uniform(ratio[0], ratio[1]))
        x = np.random.randint(0, width - patch_width)
        y = np.random.randint(0, height - patch_height)

        image[y:y + patch_height, x:x + patch_width] = 0
        mask[y:y + patch_height, x:x + patch_width] = 1

    delidx = []
    for i in range(len(target['boxes'])):
        x1, y1, x2, y2 = target['boxes'][i]
        target_area = (x2 - x1) * (y2 - y1)
        cutout_area = np.sum(mask[int(y1):int(y2), int(x1):int(x2)])
        if cutout_area / target_area >= thr:
            delidx.append(0)
        else:
            delidx.append(1)
    target = deletebox(target, delidx)

    return Image.fromarray(image), target


def pad_to_match(image, bbox, target_size):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    scale_x = target_width / original_width
    scale_y = target_height / original_height

    scale = min(scale_x, scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))

    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    resized_bbox = []
    for box in bbox['boxes']:
        x_min, y_min, x_max, y_max = box
        resized_box = [
            int(x_min * scale + left),
            int(y_min * scale + top),
            int(x_max * scale + left),
            int(y_max * scale + top)
        ]
        resized_bbox.append(resized_box)
    bbox['boxes'] = torch.tensor(resized_bbox)
    return padded_image, bbox


def merge(t1, t2):
    t1['boxes'] = torch.cat([t1['boxes'], t2['boxes']], dim=0)
    if len(t1['boxes']) == 0:
        t1['boxes'] = torch.empty((0, 4))
    t1['labels'] = torch.cat([t1['labels'], t2['labels']], dim=0)
    t1['scores'] = torch.cat([t1['scores'], t2['scores']], dim=0)
    t1['iscrowd'] = torch.cat([t1['iscrowd'], t2['iscrowd']], dim=0)
    return t1


def fix_score(label, coef):
    return label


def pseudo_mixup_with_reference(img1, img2, label1, label2, lambda_val=0.5):
    img1 = np.array(img1)
    img2 = np.array(img2)
    if lambda_val is None:
        lambda_val = np.random.beta(0.5, 0.5)

    if random.random() > 0.5:
        reference_img = img1
        reference_label = label1
        img_to_adjust = img2
        label_to_adjust = label2
    else:
        reference_img = img2
        reference_label = label2
        img_to_adjust = img1
        label_to_adjust = label1

    adjusted_img, adjusted_label = pad_to_match(img_to_adjust, label_to_adjust, reference_img.shape[1::-1])

    mixed_img = lambda_val * reference_img + (1 - lambda_val) * adjusted_img
    reference_label = fix_score(reference_label, lambda_val)
    adjusted_label = fix_score(adjusted_label, 1 - lambda_val)

    mixed_label = merge(reference_label, adjusted_label)

    return mixed_img.astype(np.uint8), mixed_label


def resize_and_pad_image(image, target_size=(400, 400), position='bottom_left'):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    pad_top = 0
    pad_left = 0

    resized_image = cv2.resize(image, (new_w, new_h))

    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    if position == 'top_left':
        pad_top = target_h - new_h
        pad_left = target_w - new_w
        padded_image[pad_top:, pad_left:] = resized_image
    elif position == 'top_right':
        pad_top = target_h - new_h
        pad_left = 0
        padded_image[pad_top:, :new_w] = resized_image
    elif position == 'bottom_left':
        pad_top = 0
        pad_left = target_w - new_w
        padded_image[:new_h, pad_left:] = resized_image
    elif position == 'bottom_right':
        pad_top = 0
        pad_left = 0
        padded_image[:new_h, :new_w] = resized_image

    return padded_image, pad_left, pad_top, scale


def adjust_bboxes(bboxes, pad_left, pad_top, scale):
    new_bboxes = []
    for bbox in bboxes['boxes']:
        x_min, y_min, x_max, y_max = bbox
        new_bbox = [
            x_min * scale + pad_left,
            y_min * scale + pad_top,
            x_max * scale + pad_left,
            y_max * scale + pad_top
        ]
        new_bboxes.append(new_bbox)
    if len(new_bboxes) == 0:
        bboxes['boxes'] = torch.empty((0, 4))
    else:
        bboxes['boxes'] = torch.tensor(new_bboxes)
    return bboxes


def pseudo_mosaic(sample):
    sampled_indices = random.sample(range(len(sample)), 4)
    sampled_images = [np.array(sample[i][0]) for i in sampled_indices]
    sampled_labels = [sample[i][1] for i in sampled_indices]

    y = random.randint(200, 400)
    x = random.randint(200, 400)
    mosaic_image = np.zeros((y * 2, x * 2, 3), dtype=np.uint8)
    mosaic_labels = None

    positions = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
    pos_coordinates = [(0, 0), (0, x), (y, 0), (y, x)]

    for i, (img, label) in enumerate(zip(sampled_images, sampled_labels)):
        padded_image, pad_left, pad_top, scale = resize_and_pad_image(img, target_size=(x, y), position=positions[i])

        adjusted_bboxes = adjust_bboxes(label, pad_left, pad_top, scale)

        pos_y, pos_x = pos_coordinates[i]
        mosaic_image[pos_y:pos_y + y, pos_x:pos_x + x] = padded_image

        correct_box = []
        for bbox in adjusted_bboxes['boxes']:
            new_bbox = [
                bbox[0] + pos_x,  # x_min
                bbox[1] + pos_y,  # y_min
                bbox[2] + pos_x,  # x_max
                bbox[3] + pos_y  # y_max
            ]
            correct_box.append(new_bbox)
        adjusted_bboxes['boxes'] = torch.tensor(correct_box)
        if mosaic_labels is None:
            mosaic_labels = adjusted_bboxes
        else:
            mosaic_labels = merge(mosaic_labels, adjusted_bboxes)

    return mosaic_image, mosaic_labels


def filter_boxes(tmp):
    valid_indices = []
    for i, box in enumerate(tmp['boxes']):
        x1, y1, x2, y2 = box
        if x2 > x1 + 1 and y2 > y1 + 1:
            valid_indices.append(i)
    tmp['boxes'] = tmp['boxes'][valid_indices]
    tmp['labels'] = tmp['labels'][valid_indices]
    if 'scores' in tmp:
        tmp['scores'] = tmp['scores'][valid_indices]
    tmp['iscrowd'] = tmp['iscrowd'][valid_indices]
    return tmp


class MixPL(object):
    def __init__(self):
        self.store_cache = []
        self.cachesize = 8

    def ismosaic(self, transpose, mosaic_idxs):
        newsample = deepcopy([self.store_cache[i] for i in mosaic_idxs])
        img, target = pseudo_mosaic(newsample)
        target = filter_boxes(target)

        target_backup = deepcopy(target)
        ret_forFRCNN = (img.shape[:2], target_backup, Image.fromarray(img))
        img = Image.fromarray(img)
        ret = transpose(img, target)
        return ret, ret_forFRCNN

    def mix(self, trans):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]) # image shape CHW
        transpose = T.Compose([
            normalize
        ])
        mixup = []
        mixup_forFRCNN = []
        moasic = None
        moasic_fasterrcnn = None
        trans_use1 = deepcopy(trans)
        last_use = deepcopy(self.store_cache)
        mixup_idxs = np.random.choice(range(4), 4, replace=False)
        mosaic_idxs = np.random.choice(range(4), 4, replace=False) + 4
        if len(last_use) >= self.cachesize:
            mn = 1
            for i in range(len(trans)):
                if i >= len(last_use) or last_use is None:
                    continue
                ckimg, ckpl = pseudo_mixup_with_reference(trans[i][0], last_use[mixup_idxs[i]][0], trans[i][1],
                                                          last_use[mixup_idxs[i]][1])
                mn = min(mn, len(ckpl['scores']))
                ckpl = filter_boxes(ckpl)
                ckpl_backup = deepcopy(ckpl)
                mixup_forFRCNN.append((ckimg.shape[:2], ckpl_backup, Image.fromarray(ckimg)))
                ckimg = Image.fromarray(ckimg)
                mixup.append(transpose(ckimg, ckpl))
            moasic, moasic_fasterrcnn = self.ismosaic(transpose, mosaic_idxs)
            if mn == 0:
                mixup = []
                mixup_forFRCNN = []
            if len(moasic[1]['labels']) == 0:
                moasic = None
                moasic_fasterrcnn = None
        self.store_cache.extend(trans_use1)
        self.store_cache = self.store_cache[-self.cachesize:]
        return mixup, moasic, mixup_forFRCNN, moasic_fasterrcnn


def prepocess(ret_forFRCNN):
    if ret_forFRCNN[0] is None:
        return [None]
    for item in ret_forFRCNN:
        data_dict = item[1]
        if "labels" in data_dict:
            data_dict["labels"] = data_dict["labels"] + 1
    return ret_forFRCNN


def strong_aug(img, target, mixpl):
    scale = [480, 512, 544, 576, 608, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088, 1120, 1152, 1184]
    transpose_img = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scale, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(400, 550),
                T.RandomResize(scale, max_size=1333),
            ])
        )
    ])
    trans = []
    for i in range(len(img)):
        img_t, target_t = transpose_img(img[i], target[i])
        img_t, target_t = rand_geometric(img_t, target_t)
        img_t = randomColor(img_t)
        img_t, target_t = cutout(img_t, target_t)
        trans.append((img_t, target_t))
    if trans:
        mpl, moasic, mixup_forfaterRCNN, moasic_forfaterRCNN = mixpl.mix(trans)
        if len(mpl):
            # image, target_ret = misc.collate_fn(mpl)
            return mpl, moasic, prepocess(mixup_forfaterRCNN), prepocess([moasic_forfaterRCNN])[0]
        else:
            return None, moasic, [], prepocess([moasic_forfaterRCNN])[0]
    else:
        return None, None, [], None


def plot_results(pil, prob, boxes):
    # pil_img = Image.open(pil)
    plt.figure(figsize=(16, 10))
    plt.imshow(pil)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig("xx.png")
    plt.show()


def compute_metrics(boxes, pseudo, coef=0.5, mincoef=0.1):
    ious = []
    average_boxes = []
    assert boxes.shape[0] == pseudo['boxes'].shape[0], "pred_boxes_batch is wrong"
    for i in range(boxes.shape[0]):
        assert pseudo['labels'][i] > 0, "pseudo['labels'][i] is 0, make sure input is target_fasterrcnn"
        x_box = boxes[i, pseudo['labels'][i]]
        y_box = pseudo['boxes'][i]
        if x_box[0] + 1 >= x_box[2] or x_box[1] + 1 >= x_box[3]:
            average_boxes.append([y_box[0], y_box[1], y_box[2], y_box[3]])
            ious.append(torch.tensor(mincoef, device=y_box.device, dtype=torch.float32))
            continue
        xi1 = max(x_box[0], y_box[0])
        yi1 = max(x_box[1], y_box[1])
        xi2 = min(x_box[2], y_box[2])
        yi2 = min(x_box[3], y_box[3])

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        area_x = (x_box[2] - x_box[0]) * (x_box[3] - x_box[1])
        area_y = (y_box[2] - y_box[0]) * (y_box[3] - y_box[1])
        union_area = area_x + area_y - inter_area

        iou = inter_area / union_area if union_area != 0 else torch.tensor(mincoef, device=y_box.device, dtype=torch.float32)
        iou = torch.clamp(iou, min=mincoef, max=1.0)
        ious.append(iou)

        avg_box = [
            coef * x_box[0] + (1 - coef) * y_box[0],
            coef * x_box[1] + (1 - coef) * y_box[1],
            coef * x_box[2] + (1 - coef) * y_box[2],
            coef * x_box[3] + (1 - coef) * y_box[3]
        ]
        average_boxes.append(avg_box)

    mean_iou = sum(ious).item() / len(ious) if len(ious) > 0 else 0.0

    return ious, mean_iou, average_boxes


def clip_boxes_torch(X, size):
    height, width = size

    X[..., 0] = torch.clamp(X[..., 0], min=0, max=width)   # x1
    X[..., 1] = torch.clamp(X[..., 1], min=0, max=height)  # y1
    X[..., 2] = torch.clamp(X[..., 2], min=0, max=width)   # x2
    X[..., 3] = torch.clamp(X[..., 3], min=0, max=height)  # y2

    assert torch.all(X[..., 0] <= X[..., 2]), "Error: Some boxes have x1 > x2 after clipping!"
    assert torch.all(X[..., 1] <= X[..., 3]), "Error: Some boxes have y1 > y2 after clipping!"
    return X


def targetFrcnn2Dino(targets, sizes):
    def NormperT(target, size):
        h, w = size
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)
            target["boxes"] = boxes
        target["labels"] = target["labels"] - 1
        return target
    targets = deepcopy(targets)
    for i, (target, size) in enumerate(zip(targets, sizes)):
        targets[i] = NormperT(target, size)
    return targets


def RefinePseudo(targets_ori, model, features, image_shapes_ori, mincoef=0.1):
    targets = deepcopy(targets_ori)
    pseudo = deepcopy(targets_ori)
    notempty = []
    for i, t in enumerate(targets):
        if len(t["boxes"]) > 0:
            notempty.append(i)
    if len(notempty) == 0:
        return targets_ori
    pseudo = [pseudo[i] for i in notempty]
    image_shapes = [image_shapes_ori[i] for i in notempty]
    names = ['0', '1', '2', '3', 'pool']
    newfeatures = [v[notempty] for k, v in features.items()]
    newfeatures = OrderedDict([(k, v) for k, v in zip(names, newfeatures)])
    box_features = model.fasterrcnn_ori.roi_heads.box_roi_pool(newfeatures, [i['boxes'] for i in pseudo], image_shapes)
    box_features = model.fasterrcnn_ori.roi_heads.box_head(box_features)
    _, box_regression = model.fasterrcnn_ori.roi_heads.box_predictor(box_features)
    pred_boxes = model.fasterrcnn_ori.roi_heads.box_coder.decode(box_regression, [i['boxes'] for i in pseudo])
    start_index = 0
    pred_boxes_batch = []
    for i in pseudo:
        pred_boxes_batch.append(pred_boxes[start_index:start_index + i['boxes'].shape[0]])
        start_index += i['boxes'].shape[0]
    assert start_index == pred_boxes.shape[0]
    for i, (boxes, pseudoboxes, image_shape) in enumerate(zip(pred_boxes_batch, pseudo, image_shapes)):
        boxes = clip_boxes_torch(boxes, image_shape)
        ious, mean_iou, average_boxes = compute_metrics(boxes, pseudoboxes, mincoef=mincoef)
        ious = torch.tensor(ious, device=pseudo[i]['boxes'].device)
        pseudo[i]['scores'] = torch.clamp(ious, min=mincoef, max=1.0)
        pseudo[i] = filter_boxes(pseudo[i])
    for i, idx in enumerate(notempty):
        targets[idx] = pseudo[i]
    return targets


def labeloffset(target, offset):
    for i in target:
        i['labels'] = i['labels'] + offset
    return target


def filter_weak_output(output, raw_strong, threshold=0.4):
    probas = output['pred_logits'].sigmoid()
    keep = probas.max(-1).values > threshold
    label = torch.argmax(probas, dim=-1)
    ans = []
    filte_emptyimage = []
    for i in range(len(keep)):
        filte_emptyimage.append(int(sum(keep[i]) > 0))
    for i in range(keep.shape[0]):
        box = rescale_bboxes(output['pred_boxes'][i, keep[i]],
                                       raw_strong[i].size)
        box = clip_boxes_torch(box, (raw_strong[i].size[1], raw_strong[i].size[0]))
        tmp = {'boxes': box,
               'labels': label[i, keep[i]],
               'scores': probas[i, keep[i]],
               'iscrowd': torch.zeros(torch.sum(keep[i]), device=output['pred_boxes'].device),
               }
        tmp = filter_boxes(tmp)
        ans.append(tmp)
    # plot_results(raw_strong[0], ans[0]['scores'],
    #              rescale_bboxes(output['pred_boxes'][0, keep[0]].cpu(),
    #                             raw_strong[0].size))
    return ans, raw_strong

