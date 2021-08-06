import numpy as np
import skimage.measure as measure


def get_three_period(mask):
    np.set_printoptions(threshold=np.inf)
    labeled_mask = measure.label(mask)
    # print(labeled_mask)
    max_label = np.max(labeled_mask)
    p_on_list = []
    p_off_list = []
    r_on_list = []
    r_off_list = []
    t_on_list = []
    t_off_list = []
    for label in range(1, max_label + 1):
        bool_array = np.where(labeled_mask == label, 1, 0)
        # print(bool_array)
        # print(np.where(bool_array))
        start, end = np.where(bool_array)[0][[0, -1]]
        # print(start, end)
        if mask[start] == 1:
            if end - start > 10:
                p_on_list.append(start)
                p_off_list.append(end)
        elif mask[start] == 2:
            if end - start > 10:
                r_on_list.append(start)
                r_off_list.append(end)
        elif mask[start] == 3:
            if end - start > 50:
                t_on_list.append(start)
                t_off_list.append(end)
    point_dict = {}
    point_dict['p_on_list'] = p_on_list
    point_dict['p_off_list'] = p_off_list
    point_dict['r_on_list'] = r_on_list
    point_dict['r_off_list'] = r_off_list
    point_dict['t_on_list'] = t_on_list
    point_dict['t_off_list'] = t_off_list
    # print(pr_period, qrs_period, t_period)
    return point_dict


if __name__ == '__main__':
    x = np.zeros((200, ))
    x[10:30] = 1
    x[40:55] = 2
    x[66:90] = 3
    x[110:140] = 1
    x[150:160] = 2
    x[180:190] = 3
    a, b, c = get_three_period(x)