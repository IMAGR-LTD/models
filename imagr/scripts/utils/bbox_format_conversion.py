

def tlwh_xyxy(bbox):
    "top left width height to xmin ymin xmax ymax"
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = xmin + bbox[2]
    ymax = ymin + bbox[3]
    return [xmin, ymin, xmax, ymax]

def cpwh_xyxy(bbox):
    "center point width height to xmin ymin xmax ymax"
    x, y, w, h = bbox
    xmin = x - w / 2
    xmax = x + w / 2
    ymin = y - h / 2 
    ymax = y + h / 2
    return [xmin, ymin, xmax, ymax]


def xyxy_cpwh(bbox):
    "bbox is a list contains xmin,ymin,xmax,ymax"
    xmin, ymin, xmax, ymax = bbox
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) / 2
    h = (ymax - ymin) / 2
    return [x, y, w, h]


def xyxy_tlwh(bbox):
    "bbox is a list contains xmin,ymin,xmax,ymax"
    xmin, ymin, xmax, ymax = bbox
    w = (xmax - xmin) / 2
    h = (ymax - ymin) / 2
    return [xmin, ymin, w, h]


