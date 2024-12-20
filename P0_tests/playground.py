import torch
from ultralytics.utils.metrics import loc_dor, loc_dor_pw, bbox_iou
from torchvision.ops.boxes import box_iou
from torchvision.ops import nms


def split_pred_euclid(gt_locs: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    n_anchors = predictions.shape[0]
    bs, n_gt, _ = gt_locs.shape
    gt_locs_reordered = gt_locs.view(-1, 1, 2)
    tensor_list = []

    fac = predictions.shape[1] // 2

    for i in range(fac):
        # after **2: (b_s*gt, n_anchors, 2)
        # after sum: (b_s*gt, n_anchors)
        euclid_dist = torch.sqrt(((predictions[None][:,:, i*2:(i+1)*2] - gt_locs_reordered) ** 2).sum(dim=2))
        tensor_list.append(euclid_dist)

    
    distances = torch.stack(tensor_list, dim=2)

    return distances.view(bs, n_gt, n_anchors, fac)



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


def nms_gpt_vec2(dets: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    if dets.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=dets.device)

    _, order = scores.sort(dim=0, descending=True)
    ndets = dets.shape[0]
    suppress_ind = torch.zeros(ndets, dtype=torch.bool, device=dets.device)
    
    bxs_ordered = dets[order]
    iou_matrix = box_iou(bxs_ordered, bxs_ordered)

    suppress_m = (iou_matrix > iou_thres).triu(diagonal=1)

    for i in range(ndets):
        if suppress_ind[i] == 1:
            continue

        suppress_ind[suppress_m[i].nonzero()] = True

    keep_ind = (~suppress_ind).nonzero()
    if keep_ind.numel() > 1:
        keep_ind.squeeze_(1)
    else:
        print("BP")

    return order[keep_ind]

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
    #### TEST WITH ONLY ONE POINT PER ANCHOR 
    preds = torch.tensor([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                          [4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
                          [100.0, 100.0, 100.0, 100.0, 100.0, .0]])
    fac = preds.shape[1] // 2

    gt_locs = torch.tensor([[[1.0, 1.0],
                             [2.0, 2.0]],
                            [[3.0, 3.0],
                             [4.0, 4.0]]])
    
    gt_radii = torch.Tensor([[[2.0], 
                              [1.0]], 
                             [[1.0], 
                              [2.0]]])
    
    dist_it = split_pred_euclid(gt_locs=gt_locs, predictions=preds)

    test = gt_radii.unsqueeze(3).expand(dist_it.shape[0], dist_it.shape[1], dist_it.shape[2], fac)

    test_mask = dist_it.le(test) # (bs, n_gts, n_anchors, n_preds_per_anchor)

    final = torch.any(test_mask, dim=3)

    print("BP")


    """
    ####################################### NMS #################################################################
    #############################################################################################################
    bxs = torch.randint(10,(2,4))
    scores = torch.randint(10, (2,1)).squeeze()
    iou_thres = 0.7


    nms_gpt_vec(bxs, scores, iou_thres)
    nms_gpt(bxs, scores, iou_thres)

    ####################################### DOR #################################################################
    #############################################################################################################
    n_loc1 = 2
    n_loc2 = 3

    
    loc1 = torch.randint(11, (n_loc1, 2)).float()
    loc2 = torch.randint(11, (n_loc2, 2)).float()
    radii = torch.randint(5, (n_loc1,1))

    test_loc = loc_dor_pw(loc1=loc1, loc2=loc2, radii=radii)

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