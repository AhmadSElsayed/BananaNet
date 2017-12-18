import cv2
import numpy
import utility

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def get_landmarks(im, points):
    #assuming the order of points is left eye, right eye, nose, mouth left, mouth right
    left_eye=points[0]
    right_eye=points[1]
    nose=points[2]
    mouth_left = points[3]
    mouth_right = points[4]
    eye_dist= numpy.linalg.norm( numpy.array(left_eye) - numpy.array(right_eye) )
    eye_factor=int(0.25*eye_dist // 1)
    mouth_dist= numpy.linalg.norm( numpy.array(mouth_left) - numpy.array(mouth_right) )
    mouth_factor=int(0.25*mouth_dist // 1)
    left_eye=(-1 * eye_factor + left_eye[0], -1 * eye_factor + left_eye[1])
    right_eye = (1 * eye_factor + right_eye[0], -1 * eye_factor + right_eye[1])
    mouth_left=(-1 * mouth_factor + mouth_left[0], 1 * mouth_factor + mouth_left[1])
    mouth_right=(1 * mouth_factor + mouth_right[0], 1 * mouth_factor + mouth_right[1])
    points=numpy.array([left_eye, right_eye, nose, mouth_left, mouth_right])
    return numpy.matrix(points)


def draw_convex_hull(im, points, color):
	points = cv2.convexHull(points)
	cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    draw_convex_hull(im, landmarks, color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def warpImage(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    dist=landmarks1[0] - landmarks1[1]
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(numpy.array(dist[0,0], dist[0,1]))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (255 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    img=(im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))

    #utility.show_images([img, im1_blur, im2_blur])
    return img


sourceImage = ""
sourceLandmarks = ""
mask = ""

def initSwappingModule(swapImageSource, points):
    global sourceImage
    sourceImage = swapImageSource.copy()
    global sourceLandmarks
    sourceLandmarks = get_landmarks(sourceImage, points)
    global mask
    mask = get_face_mask(sourceImage, sourceLandmarks)


def swap (destImage, points):
    global mask
    destLandmarks = get_landmarks(destImage, points)

    M = transformation_from_points(destLandmarks, sourceLandmarks)

    warped_mask = warpImage(mask, M, destImage.shape)
    combined_mask = numpy.max([get_face_mask(destImage, destLandmarks), warped_mask],
                              axis=0)

    warpedSourceImage = warpImage(sourceImage, M, destImage.shape)
    warpedCorrectedSource = correct_colours(destImage, warpedSourceImage, destLandmarks)

    warpedCorrectedSource=warpedSourceImage
    warpedCorrectedSource=warpedCorrectedSource.astype(numpy.uint8)
    result = destImage * (1.0 - combined_mask) + warpedCorrectedSource * combined_mask
    #utility.show_images([mask,warped_mask,combined_mask,warpedSourceImage,warpedCorrectedSource,result])
    #cv2.imwrite('output.jpg', result)
    return result.astype(numpy.uint8)
