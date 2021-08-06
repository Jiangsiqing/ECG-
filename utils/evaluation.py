import numpy as np


def do_evaluation(pred, gt):
    np.set_printoptions(threshold=np.inf)
    pred = np.array(pred)
    gt = np.array(gt)
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    # print(pred)
    intersection_array = np.where(pred == gt, 1, 0)
    intersection_array = (pred > 0) * intersection_array * (gt > 0)
    intersection = np.sum(intersection_array)
    union = np.sum(pred > 0) + np.sum(gt > 0)
    dice = (2. * intersection) / (union + 1e-6)
    result = {
        'dice': dice
    }
    return result


def do_evaluation_l1(pred_dict, gt_dict):
    if len(pred_dict['t_start']) != len(gt_dict['t_start']):
        return None, None
    class_l1 = np.zeros((5,))
    class_recorder = np.zeros((5, ))
    for index, (pred_list, gt_list) in enumerate(zip(pred_dict.values(), gt_dict.values())):
        if np.sum(abs(np.array(pred_list) - np.array(gt_list))) > 100:
            return None, None
        class_l1[index] += np.sum(abs(np.array(pred_list) - np.array(gt_list)))
        class_recorder[index] += len(pred_list)
    return class_l1, class_recorder


if __name__ == '__main__':

    pred = np.random.randint(0, 100, (4, 1280)) % 4
    gt = np.random.randint(0, 100, (4, 1280)) % 4
