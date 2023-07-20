import cv2 
import numpy as np
from settings.facial import DEFAULT_TRIANGULATION
from landmarks import add_boundary_points


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src=src,
                          M=warp_mat,
                          dsize=(size[0], size[1]),
                          dst=None,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def warp_triangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(img=mask,
                       points=np.int32(t2_rect_int),
                       color=(1.0, 1.0, 1.0),
                       lineType=cv2.LINE_AA,
                       shift=0)

    # Apply warpImage to small rectangular patches
    img1_rect = np.array(img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]])
    if len(img1_rect) > 0:
        rect_shape = (r2[2], r2[3])
        img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, rect_shape)
        img2_rect = img2_rect * mask
        try:
            # Copy triangular region of the rectangular patch to the output image
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                        (1.0, 1.0, 1.0) - mask)
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect
        except ValueError:
            pass # TODO: Fix this


def warp_all(input_image, input_triangles, new_triangles):
    if type(input_image) == np.ndarray:
        input_image_warped = input_image.copy()
    else:
        input_image_warped = input_image.numpy().copy()

    for current_ipunt_triangle, current_new_triangle in zip(input_triangles,new_triangles):
        warp_triangle(input_image, input_image_warped, current_ipunt_triangle, current_new_triangle)

    return input_image_warped

def get_new_landmarks(source_landmarks_previous, source_landmarks_current, target_landmarks_current):
    return np.add(np.subtract(source_landmarks_current, source_landmarks_previous), target_landmarks_current)


def deform(input_image, input_landmarks, previous_source_landmarks, current_source_landmarks):
    input_landmarks = add_boundary_points(input_landmarks, input_image.shape[0], input_image.shape[1])
    from scipy.spatial import Delaunay
    input_landmarks = np.array(input_landmarks)
    tri = Delaunay(input_landmarks)
    print(tri.simplices)
    raise Exception("STOP")
    previous_source_landmarks = add_boundary_points(previous_source_landmarks, input_image.shape[0], input_image.shape[1])
    current_source_landmarks = add_boundary_points(current_source_landmarks, input_image.shape[0], input_image.shape[1])
    input_triangles = input_landmarks[DEFAULT_TRIANGULATION]
    new_landmarks = get_new_landmarks(previous_source_landmarks, current_source_landmarks, input_landmarks)
    new_triangles = new_landmarks[DEFAULT_TRIANGULATION]
    return warp_all(input_image, input_triangles, new_triangles), new_landmarks
