import cv2
from skimage.metrics import structural_similarity as ssim

# TODO add contour detection for enhanced accuracy


def match(path1, path2):
    # read the images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    # turn images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))
    # display both images
    cv2.imshow("One", img1)
    cv2.imshow("Two", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    similarity_value = ssim(img1, img2) * 100

    # Define a threshold value for similarity
    threshold = 75  # You can adjust this value based on your needs

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


# Example usage
# ans = match("D:\\Code\\Git stuff\\Signature-Matching\\assets\\1.png",
# "D:\\Code\\Git stuff\\Signature-Matching\\assets\\3.png")
