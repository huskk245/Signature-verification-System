import cv2
from skimage.metrics import structural_similarity as ssim


def match(path1, path2):
    # read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # turn images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # resize images for comparison
    img1_gray = cv2.resize(img1_gray, (300, 300))
    img2_gray = cv2.resize(img2_gray, (300, 300))

    # Apply GaussianBlur to reduce noise
    img1_gray = cv2.GaussianBlur(img1_gray, (5, 5), 0)
    img2_gray = cv2.GaussianBlur(img2_gray, (5, 5), 0)

    # Detect contours in the images
    _, thresh1 = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original images for visualization
    img1_contour = cv2.drawContours(img1.copy(), contours1, -1, (0, 255, 0), 3)
    img2_contour = cv2.drawContours(img2.copy(), contours2, -1, (0, 255, 0), 3)

    # Display both images with contours
    cv2.imshow("One with Contour", img1_contour)
    cv2.imshow("Two with Contour", img2_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    similarity_value = ssim(img1_gray, img2_gray) * 100

    # Define a threshold value for similarity
    threshold = 85  # can adjust this value based on your needs

    if similarity_value >= threshold:
        print(
            "The signature is likely to be real with a similarity value of {:.2f}%".format(
                similarity_value
            )
        )
    else:
        print(
            "The signature is likely to be fake with a similarity value of {:.2f}%".format(
                similarity_value
            )
        )

    return similarity_value
