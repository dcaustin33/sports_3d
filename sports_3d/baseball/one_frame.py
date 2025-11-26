import cv2

def read_bbox(bbox_path):
    with open(bbox_path, 'r') as f:
        lines = f.readlines()
    final = []
    for line in lines:
        ind_line = []
        for item in line.strip().split()[1:]:
            ind_line.append(float(item))
        final.append(ind_line)
    return final[0]

if __name__ == "__main__":
    # image_path = "/Users/derek/Desktop/sports_3d/data/pitch/frame_005776_t96.363s.png"
    # image = cv2.imread(image_path)
    # bbox_path = "/Users/derek/Desktop/sports_3d/data/pitch/frame_005776_t96.363s_bbox.txt"
    # bbox = read_bbox(bbox_path)
    #
    # distance_to_ball = 18.44
    # ball_width_m = 0.073
    # image_width = image.shape[1]
    # print(image.shape)
    # print(bbox)
    # focal = (image_width * bbox[2] * distance_to_ball) / ball_width_m
    # print(focal)
    pass