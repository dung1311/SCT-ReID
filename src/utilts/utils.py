def calc_box_probs(box):
    x0, y0, x2, y2 = box[:4]
    w = x2-x0
    h = y2-y0
    area = w*h
    r = h/(1.0*w)
    return w, h, r, area

def select_boxes(batch_boxes, cfg_select):
    boxes_ret = []
    for box in batch_boxes:
        w, h, r, a = calc_box_probs(box)
        x0, y0, x2, y2 = box
        x_c, y_c = (x0 + x2)/2, y2
        
        if x_c < cfg_select["roi_det"][0] or x_c > cfg_select["roi_det"][2] or \
        y_c < cfg_select["roi_det"][1] or y_c > cfg_select["roi_det"][3] or \
        w<cfg_select["w_min"] or h<cfg_select["h_min"] or \
        a<cfg_select["a_min"] or \
        r<cfg_select["h_w_r_min"] or r>cfg_select["h_w_r_max"]:
            continue
        boxes_ret.append(box)
        
    return boxes_ret