import os

import nibabel as nib
import numpy as np
import torch
import csv
import utils.common_utils as common_utils
import utils.data_utils as du


def dice_confusion_matrix(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_perclass = torch.zeros(num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def compute_volume(prediction_map, label_map, ID):
    num_cls = len(label_map) - 1
    volume_dict = {}
    volume_dict['vol_ID'] = ID
    for i in range(num_cls):
        binarized_pred = (prediction_map == i).astype(float)
        volume_dict[label_map[i + 1]] = np.sum(binarized_pred)

    return volume_dict


def evaluate_dice_score(model_path, num_classes, data_dir, label_dir, volumes_txt_file, remap_config, orientation,
                        prediction_path, device=0, logWriter=None, mode='eval'):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

    batch_size = 20

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    model = torch.load(model_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model.cuda(device)

    model.eval()

    common_utils.create_if_not(prediction_path)
    volume_dice_score_list = []
    print("Evaluating now...")
    file_paths = du.load_file_paths(data_dir, label_dir, volumes_txt_file)
    with torch.no_grad():
        for vol_idx, file_path in enumerate(file_paths):
            volume, labelmap, class_weights, weights, header = du.load_and_preprocess(file_path,
                                                                                      orientation=orientation,
                                                                                      remap_config=remap_config)

            volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
            volume, labelmap = torch.tensor(volume).type(torch.FloatTensor), torch.tensor(labelmap).type(
                torch.LongTensor)

            volume_prediction = []
            for i in range(0, len(volume), batch_size):
                batch_x, batch_y = volume[i: i + batch_size], labelmap[i:i + batch_size]
                if cuda_available:
                    batch_x = batch_x.cuda(device)
                out = model(batch_x)
                _, batch_output = torch.max(out, dim=1)
                volume_prediction.append(batch_output)

            volume_prediction = torch.cat(volume_prediction)
            volume_dice_score = dice_score_perclass(volume_prediction, labelmap.cuda(device), num_classes, mode=mode)

            volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
            nifti_img = nib.MGHImage(np.squeeze(volume_prediction), np.eye(4), header=header)
            nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('.mgz')))
            if logWriter:
                logWriter.plot_dice_score('val', 'eval_dice_score', volume_dice_score, volumes_to_use[vol_idx], vol_idx)

            volume_dice_score = volume_dice_score.cpu().numpy()
            volume_dice_score_list.append(volume_dice_score)
            print(volume_dice_score, np.mean(volume_dice_score))
        dice_score_arr = np.asarray(volume_dice_score_list)
        avg_dice_score = np.mean(dice_score_arr)
        print("Mean of dice score : " + str(avg_dice_score))
        class_dist = [dice_score_arr[:, c] for c in range(num_classes)]

        if logWriter:
            logWriter.plot_eval_box_plot('eval_dice_score_box_plot', class_dist, 'Box plot Dice Score')
    print("DONE")

    return avg_dice_score, class_dist


def _segment_vol(file_path, model, orientation, batch_size, cuda_available, device):
    volume, header = du.load_and_preprocess_eval(file_path,
                                                 orientation=orientation)

    volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
    volume = torch.tensor(volume).type(torch.FloatTensor)

    volume_pred = []
    for i in range(0, len(volume), batch_size):
        batch_x = volume[i: i + batch_size]
        if cuda_available:
            batch_x = batch_x.cuda(device)
        out = model(batch_x)
        # _, batch_output = torch.max(out, dim=1)
        volume_pred.append(out)

    volume_pred = torch.cat(volume_pred)
    _, volume_prediction = torch.max(volume_pred, dim=1)

    volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
    volume_prediction = np.squeeze(volume_prediction)
    if orientation == "COR":
        volume_prediction = volume_prediction.transpose((1, 2, 0))
        volume_pred = volume_pred.permute((2, 1, 3, 0))
    elif orientation == "AXI":
        volume_prediction = volume_prediction.transpose((2, 0, 1))
        volume_pred = volume_pred.permute((3, 1, 0, 2))


    return volume_pred, volume_prediction, header


def evaluate(coronal_model_path, volumes_txt_file, data_dir, device, prediction_path, batch_size, orientation,
             label_names):
    print("**Starting evaluation**")
    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    model = torch.load(coronal_model_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model.cuda(device)

    model.eval()

    common_utils.create_if_not(prediction_path)
    print("Evaluating now...")
    file_paths = du.load_file_paths_eval(data_dir, volumes_txt_file)

    with torch.no_grad():
        volume_dict_list = []
        for vol_idx, file_path in enumerate(file_paths):
            _, volume_prediction, header = _segment_vol(file_path, model, orientation, batch_size, cuda_available,
                                                        device)

            nifti_img = nib.MGHImage(volume_prediction, np.eye(4), header=header)
            nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('.mgz')))
            per_volume_dict = compute_volume(volume_prediction, label_names, volumes_to_use[vol_idx])
            volume_dict_list.append(per_volume_dict)

        file_name = 'volume_estimates.csv'
        file_path = os.path.join(prediction_path, file_name)
        # Save volume_dict as csv file in the prediction_path
        with open(file_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=label_names)
            writer.writeheader()

            for data in volume_dict_list:
                writer.writerow(data)

    print("DONE")


def evaluate2view(coronal_model_path, axial_model_path, volumes_txt_file, data_dir, device, prediction_path, batch_size,
                  label_names):
    print("**Starting evaluation**")
    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    model1 = torch.load(coronal_model_path)
    model2 = torch.load(axial_model_path)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model1.cuda(device)
        model2.cuda(device)

    model1.eval()
    model2.eval()

    common_utils.create_if_not(prediction_path)
    print("Evaluating now...")
    file_paths = du.load_file_paths_eval(data_dir, volumes_txt_file)

    with torch.no_grad():
        volume_dict_list = []
        for vol_idx, file_path in enumerate(file_paths):
            volume_prediction_cor, _, header = _segment_vol(file_path, model1, "COR", batch_size, cuda_available,
                                                            device)
            volume_prediction_axi, _, header = _segment_vol(file_path, model2, "AXI", batch_size, cuda_available,
                                                            device)
            _, volume_prediction = torch.max(volume_prediction_axi + volume_prediction_cor, dim=1)
            volume_prediction = (volume_prediction.cpu().numpy()).astype('float32')
            volume_prediction = np.squeeze(volume_prediction)
            nifti_img = nib.MGHImage(volume_prediction, np.eye(4), header=header)
            nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('.mgz')))

            # nifti_img = nib.MGHImage(temp_cor, np.eye(4), header=header)
            # nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('_cor.mgz')))
            #
            # nifti_img = nib.MGHImage(temp_ax, np.eye(4), header=header)
            # nib.save(nifti_img, os.path.join(prediction_path, volumes_to_use[vol_idx] + str('_ax.mgz')))

            per_volume_dict = compute_volume(volume_prediction, label_names, volumes_to_use[vol_idx])
            volume_dict_list.append(per_volume_dict)

        file_name = 'volume_estimates.csv'
        file_path = os.path.join(prediction_path, file_name)
        # Save volume_dict as csv file in the prediction_path
        with open(file_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=label_names)
            writer.writeheader()

            for data in volume_dict_list:
                writer.writerow(data)

    print("DONE")
