import torch
from ultralytics.utils.metrics import loc_dor, loc_dor_pw, bbox_iou
from torchvision.ops.boxes import box_iou

def nms_tut(bboxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    order = torch.argsort(-scores).to("cuda")
    indices = torch.arange(bboxes.shape[0], device="cuda")
    keep = torch.ones_like(indices, dtype=torch.bool, device="cuda")
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > iou_threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]

def nms_gpt(dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:

    if dets.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=dets.device)

    x1_t = dets[:, 0]
    y1_t = dets[:, 1]
    x2_t = dets[:, 2]
    y2_t = dets[:, 3]

    areas_t = (x2_t - x1_t) * (y2_t - y1_t)

    _, order_t = scores.sort(dim=0, descending=True)
    ndets = dets.size(0)
    suppressed_t = torch.zeros(ndets, dtype=torch.uint8, device=dets.device)
    keep_t = torch.zeros(ndets, dtype=torch.long, device=dets.device)

    num_to_keep = 0

    for _i in range(ndets):
        i = order_t[_i]
        if suppressed_t[i] == 1:
            continue
        keep_t[num_to_keep] = i
        num_to_keep += 1
        ix1 = x1_t[i]
        iy1 = y1_t[i]
        ix2 = x2_t[i]
        iy2 = y2_t[i]
        iarea = areas_t[i]

        for _j in range(_i + 1, ndets):
            j = order_t[_j]
            if suppressed_t[j] == 1:
                continue
            xx1 = max(ix1, x1_t[j])
            yy1 = max(iy1, y1_t[j])
            xx2 = min(ix2, x2_t[j])
            yy2 = min(iy2, y2_t[j])

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            ovr = inter / (iarea + areas_t[j] - inter)
            if ovr > iou_threshold:
                suppressed_t[j] = 1

    return keep_t[:num_to_keep]

def nms_gpt_vec(dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:

    if dets.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=dets.device)

    x1_t, y1_t, x2_t, y2_t = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas_t = (x2_t - x1_t) * (y2_t - y1_t)

    _, order_t = scores.sort(dim=0, descending=True)
    ndets = dets.size(0)

    ix1, iy1, ix2, iy2 = x1_t[order_t], y1_t[order_t], x2_t[order_t], y2_t[order_t]
    iarea = areas_t[order_t]

    xx1 = torch.maximum(ix1.unsqueeze(1), x1_t.unsqueeze(0))
    yy1 = torch.maximum(iy1.unsqueeze(1), y1_t.unsqueeze(0))
    xx2 = torch.minimum(ix2.unsqueeze(1), x2_t.unsqueeze(0))
    yy2 = torch.minimum(iy2.unsqueeze(1), y2_t.unsqueeze(0))

    w = torch.maximum(torch.zeros_like(xx2), xx2 - xx1)
    h = torch.maximum(torch.zeros_like(yy2), yy2 - yy1)
    inter = w * h
    ovr = inter / (iarea.unsqueeze(1) + areas_t.unsqueeze(0) - inter)
    
    # Initialize a mask to identify boxes to keep
    suppress_mask = torch.zeros(ndets, dtype=torch.bool, device=dets.device)

    # Loop through the IoU matrix and update suppression mask
    for i in range(ndets):
        if not suppress_mask[i]:
            suppress_mask |= (ovr[i] > iou_threshold)

    # Invert the suppression mask to get indices of boxes to keep
    keep_indices = (~suppress_mask).nonzero().squeeze()

    # Get original indices from the sorted order
    keep_t = order_t[keep_indices]

    return keep_t



def pairwise_px_dist(image_height, image_width, sample_size_1, sample_size_2):
    total_pixels = image_height * image_width
    
    # Generate random indices for sampling
    indices_1 = torch.randperm(total_pixels)[:sample_size_1]
    indices_2 = torch.randperm(total_pixels)[:sample_size_2]
    
    # Convert indices to 2D coordinates
    coordinates_1 = torch.stack([indices_1 // image_width, indices_1 % image_width], dim=1)
    coordinates_2 = torch.stack([indices_2 // image_width, indices_2 % image_width], dim=1)

    coord1_norm = torch.column_stack((coordinates_1[:, 0] / image_width, coordinates_1[:, 1] / image_height))
    coord2_norm = torch.column_stack((coordinates_2[:, 0] / image_width, coordinates_2[:, 1] / image_height))


    print(f"Sampled coord set 1:\n" \
          f"absolute:\n{coordinates_1}\n" \
          f"normalized:\n{coord1_norm}\n\n" \
          f"Sampled coord set 2:\n" \
          f"absolute:\n{coordinates_2}\n" \
          f"nomralized:\n{coord2_norm}\n\n")
    
    # Calculate pairwise distances
    distances_abs = torch.cdist(coordinates_1.float(), coordinates_2.float(), p=2)
    distances_norm = torch.cdist(coord1_norm, coord2_norm)

    distances_abs_ribera = cdist_ribera(coordinates_1.float(), coordinates_2.float())
    distances_norm_ribera = cdist_ribera(coord1_norm, coord2_norm)

    
    return distances_abs, distances_norm, distances_abs_ribera, distances_norm_ribera

def cdist_ribera(x, y):
    """
    Compute distance between each pair of the two collections of inputs. As implemented in Ribera et al.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||

    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**2, -1).sqrt()
    return distances



if __name__ == "__main__":
    ####################################### DOR #################################################################
    #############################################################################################################
    n_loc1 = 2
    n_loc2 = 3

    
    loc1 = torch.randint(11, (n_loc1, 2)).float()
    loc2 = torch.randint(11, (n_loc2, 2)).float()
    radii = torch.randint(5, (n_loc1,1))

    test_loc = loc_dor_pw(loc1=loc1, loc2=loc2, radii=radii)


    print("BP")


    """
    ####################################### LE-BROADCASTING #####################################################
    #############################################################################################################

    values = torch.randint(4, (3, 2, 5))
    filter = torch.randint(3, (3, 2, 1))

    values_filtered = values.le(filter)

    ####################################### RIBERA COMPARISON ###################################################
    #############################################################################################################
    image_height = 100  # Example image height
    image_width = 100   # Example image width
    sample_size_1 = 4  # Number of pixels in the first sample
    sample_size_2 = 4  # Number of pixels in the second sample

    
    dist_abs, dist_norm, dist_abs_ribera, dist_norm_ribera = pairwise_px_dist(image_height, image_width, sample_size_1, sample_size_2)

    assert torch.equal(dist_abs, dist_abs_ribera)
    assert torch.equal(dist_norm, dist_norm_ribera)


    ####################################### DOR #################################################################
    #############################################################################################################
    n_loc1 = 3
    n_loc2 = 3

    
    loc1 = torch.randint(11, (n_loc1, 2))
    loc2 = torch.randint(11, (n_loc2, 2))

    test_loc = loc_dor(loc1=loc1, loc2=loc2, radius=1.5)


    ####################################### IOU #################################################################
    #############################################################################################################
    n_box1 = 1
    n_box2 = 4

    
    box1 = torch.randint(11, (n_box1, 4))
    box2 = torch.randint(11, (n_box2, 4))

    test_box = bbox_iou(box1=box1, box2=box2, xywh=False)

    ####################################### CANDIDATES IN RADIUS ################################################
    #############################################################################################################
    bs = 5
    n_anchors = 20
    n_gts = 15
    anchors_test = torch.randint(11, (n_anchors, 2))
    gt_test = torch.randint(11, (bs, n_gts, 2))

    gt_view = gt_test.view(-1, 1, 2)
    diff = anchors_test[None] - gt_view
    squared = diff ** 2
    sum_squared = squared.sum(dim=2)
    sum_squared_view = sum_squared.view(bs, n_gts, n_anchors)

    ####################################### PAIRWISE PX DIST ###################################################
    ############################################################################################################
    image_height = 100  # Example image height
    image_width = 100   # Example image width
    sample_size_1 = 4  # Number of pixels in the first sample
    sample_size_2 = 4  # Number of pixels in the second sample

    distances_abs, distances_norm = pairwise_px_dist(image_height, image_width, sample_size_1, sample_size_2)

    print(f"Distances abs:\n{distances_abs}\n\nDistances norm:\n{distances_norm}")
    
    """