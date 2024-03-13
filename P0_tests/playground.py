import torch
from ultralytics.utils.metrics import loc_dor, bbox_iou

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
    ####################################### RADIUS-LABEL MAPPING#################################################
    #############################################################################################################

    radii = {0: 5, 1: 3, 2: 1, 3: 0}

    labels = torch.randint(4, (3, 2, 2))

    






    





    print("BP")


    """

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